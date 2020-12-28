import torch
import torch.nn as nn
import yaml

from src.solver import BaseSolver

from src.asr import ASR
from src.optim import Optimizer
from src.data import load_dataset
from src.util import human_format, cal_er, feat_to_fig, LabelSmoothingLoss
from src.audio import Delta, Postprocess, Augment

EMPTY_CACHE_STEP = 100
STOP_EPOCH = 10

class Solver(BaseSolver):
    ''' Solver for training'''
    def __init__(self,config,paras,mode):
        super().__init__(config,paras,mode)

        self.val_mode = self.config['hparas']['val_mode'].lower()
        self.WER = 'per' if self.val_mode == 'per' else 'wer'

        '''early stopping for ctc '''
        self.early_stoping = self.config['hparas']['early_stopping']

    def fetch_data(self, data, train=False):
        ''' Move data to device and compute text seq. length'''
        # feat: B x T x D
        _, feat, feat_len, txt = data
        feat = feat.to(self.device)
        feat_len = feat_len.to(self.device)
        txt = txt.to(self.device)
        txt_len = torch.sum(txt!=0,dim=-1)
        
        return feat, feat_len, txt, txt_len

    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.feat_dim, self.vocab_size, self.tokenizer, msg = \
                         load_dataset(self.paras.njobs, self.paras.gpu, self.paras.pin_memory, 
                                      False, **self.config['data'])
        self.verbose(msg)

        # Dev set sames
        self.dv_names = []
        if type(self.dv_set) is list:
            for ds in self.config['data']['corpus']['dev_split']:
                self.dv_names.append(ds[0])
        else:
            self.dv_names = self.config['data']['corpus']['dev_split'][0]
        
        # Logger settings
        if type(self.dv_names) is str:
            self.best_wer = {'att':{self.dv_names:3.0},
                             'ctc':{self.dv_names:3.0}}
        else:
            self.best_wer = {'att': {},'ctc': {}}
            for name in self.dv_names:
                self.best_wer['att'][name] = 3.0
                self.best_wer['ctc'][name] = 3.0

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        # Model
        #print(self.feat_dim) #160
        batch_size = self.config['data']['corpus']['batch_size']//2
        self.model = ASR(self.feat_dim, self.vocab_size, batch_size, **self.config['model']).to(self.device)



        self.verbose(self.model.create_msg())
        model_paras = [{'params':self.model.parameters()}]

        # Losses
        
        '''label smoothing'''
        if self.config['hparas']['label_smoothing']:
            self.seq_loss = LabelSmoothingLoss(31, 0.1)   
            print('[INFO]  using label smoothing. ') 
        else:    
            self.seq_loss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False) # Note: zero_infinity=False is unstable?

        # Optimizer
        def get_optimizer(parameters, optimizer, lr, eps, weight_decay=0, amsgrad=False, **kwargs):
            Opt = getattr(torch.optim, optimizer)
            return Opt(parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        self.optimizer = get_optimizer(model_paras, **self.config['hparas'])

        # Setup teacher forcing scheduler
        def get_tf_scheduler(tf_start=1, tf_end=1, tf_step=1, tf_step_start=0, **kwargs):
            return lambda step: max(tf_end,tf_start-(tf_start-tf_end)*(step-tf_step_start)/tf_step if step >= tf_step_start else 1)
        self.tf_rate = get_tf_scheduler(**self.config['hparas'])

        # Enable AMP if needed
        self.enable_apex()
        
        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

    def forward_train(self, feat, feat_len, txt, global_step):
        feat_len = feat_len.to(feat.device)
        txt = txt.to(feat.device)
        txt_len = torch.sum(txt!=0,dim=-1)

        batch_size = len(feat)
        tf_rate = self.tf_rate(global_step)
        total_loss = 0

        self.timer.cnt('rd')
        # Forward model
        # Note: txt should NOT start w/ <sos>
        ctc_output, encode_len, att_output, att_align, dec_state = \
            self.model( feat, feat_len, max(txt_len), tf_rate=tf_rate, teacher=txt)
        # Clear not used objects
        del att_align
        del dec_state

        ''' early stopping ctc'''
        if self.early_stoping:
            stop_step = len(self.tr_set) * STOP_EPOCH // batch_size
            if global_step > stop_step:
                ctc_output = None
                self.model.ctc_weight = 0
        #print(ctc_output.shape)
        # Compute all objectives
        if ctc_output is not None:
            if self.paras.cudnn_ctc:
                ctc_loss = self.ctc_loss(ctc_output.transpose(0,1),
                                         txt.to_sparse().values().to(device='cpu',dtype=torch.int32),
                                         [ctc_output.shape[1]]*len(ctc_output),
                                         #[int(encode_len.max()) for _ in encode_len],
                                         txt_len.cpu().tolist())
            else:
                ctc_loss = self.ctc_loss(ctc_output.transpose(0,1), txt, encode_len, txt_len)
            total_loss += ctc_loss*self.model.ctc_weight
            del encode_len

        if att_output is not None:
            #print(att_output.shape)
            b,t,_ = att_output.shape
            att_loss = self.seq_loss(att_output.view(b*t,-1),txt.view(-1))
            # Sum each uttr and devide by length then mean over batch
            # att_loss = torch.mean(torch.sum(att_loss.view(b,t),dim=-1)/torch.sum(txt!=0,dim=-1).float())
            total_loss += att_loss*(1-self.model.ctc_weight)

        self.timer.cnt('fw')

        # Logger
        step = global_step + 1
        if (step==1) or (step%self.PROGRESS_STEP==0):
            if att_output is not None:
                self.write_log('loss',{'tr_att':att_loss})
                self.write_log(self.WER,{'tr_att':cal_er(self.tokenizer,att_output,txt)})
                self.write_log(   'cer',{'tr_att':cal_er(self.tokenizer,att_output,txt,mode='cer')})
            if ctc_output is not None:
                self.write_log('loss',{'tr_ctc':ctc_loss})
                self.write_log(self.WER,{'tr_ctc':cal_er(self.tokenizer,ctc_output,txt,ctc=True)})
                self.write_log(   'cer',{'tr_ctc':cal_er(self.tokenizer,ctc_output,txt,mode='cer',ctc=True)})
                self.write_log('ctc_text_train',self.tokenizer.decode(ctc_output[0].argmax(dim=-1).tolist(),ignore_repeat=True))
            # if self.step==1 or self.step % (self.PROGRESS_STEP * 5) == 0:
            #     self.write_log('spec_train',feat_to_fig(feat[0].transpose(0,1).cpu().detach(), spec=True))
            #del total_loss

        lr = self.optimizer.param_groups[0]['lr']        
        if step == 1:
            print('[INFO]    using lr schedular defined by Daniel, init lr = ', lr)
        if step >99999 and step%2000==0:
            lr = lr*0.85
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('[INFO]     at step:', step)
            print('[INFO]   lr reduce to', lr)

        return total_loss
        

    def exec(self):
        ''' Training End-to-end ASR system '''
        self.verbose('Total training steps {}.'.format(human_format(self.max_step)))
        
        self.n_epochs = 0
        self.timer.set()

        while self.step< self.max_step:
            for data in self.tr_set:
                _, feat, feat_len, txt = data
                feat = feat.to(self.device)
                batch_size = len(feat)

                self.optimizer.zero_grad()
            
                # Forward
                total_loss = self.forward_train(feat, feat_len, txt, self.step)

                # Backprop
                grad_norm = self.backward(total_loss)             

                self.step+=1

                # Logging
                if (self.step==1) or (self.step%self.PROGRESS_STEP==0):
                    self.progress('Tr stat | Loss - {:.2f} | {}'\
                        .format(total_loss.cpu().item(),self.timer.show()))

                # Validation
                if (self.step==1) or (self.step%self.valid_step == 0):
                    if type(self.dv_set) is list:
                        for dv_id in range(len(self.dv_set)):
                            self.validate(self.dv_set[dv_id], self.dv_names[dv_id])
                    else:
                        self.validate(self.dv_set, self.dv_names)
                if self.step % (len(self.tr_set)// batch_size)==0: # one epoch
                    print('Have finished epoch: ', self.n_epochs)
                    self.n_epochs +=1
                    
                    #self.lr_scheduler.step(total_loss)
                # End of step
                # if self.step % EMPTY_CACHE_STEP == 0:
                    # Empty cuda cache after every fixed amount of steps
                torch.cuda.empty_cache() # https://github.com/pytorch/pytorch/issues/13246#issuecomment-529185354
                self.timer.set()
                if self.step > self.max_step: break
            
            
            
            #update lr_scheduler
            
            
        self.log.close()
        print('[INFO] Finished training after', human_format(self.max_step), 'steps.')

    def forward_validate(self, feat, feat_len, txt, global_step, batch_num, batch_id, _name):
        feat_len = feat_len.to(feat.device)
        txt = txt.to(feat.device)
        txt_len = torch.sum(txt!=0,dim=-1)
        batch_size = len(feat)

        if not hasattr(self, 'dev_wer'):
            self.dev_wer = {'att':[],'ctc':[]}
        if not hasattr(self, 'dev_cer'):
            self.dev_cer = {'att':[],'ctc':[]}
        if not hasattr(self, 'dev_er'):
            self.dev_er  = {'att':[],'ctc':[]}

        dev_wer = self.dev_wer
        dev_cer = self.dev_cer
        dev_er = self.dev_er


        # Forward model
        with torch.no_grad():
            ctc_output, encode_len, att_output, att_align, dec_state = \
                self.model( feat, feat_len, int(max(txt_len)*self.DEV_STEP_RATIO))

        if att_output is not None:
            dev_wer['att'].append(cal_er(self.tokenizer,att_output,txt,mode='wer'))
            dev_cer['att'].append(cal_er(self.tokenizer,att_output,txt,mode='cer'))
            dev_er['att'].append(cal_er(self.tokenizer,att_output,txt,mode=self.val_mode))
        if ctc_output is not None:
            dev_wer['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode='wer',ctc=True))
            dev_cer['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode='cer',ctc=True))
            dev_er['ctc'].append(cal_er(self.tokenizer,ctc_output,txt,mode=self.val_mode,ctc=True))

        # Show some example on tensorboard
        if batch_id == batch_num//2:
            for i in range(min(len(txt),self.DEV_N_EXAMPLE)):
                if self.step==1:
                    self.write_log('true_text_{}_{}'.format(_name, i),self.tokenizer.decode(txt[i].tolist()))
                if att_output is not None:
                    self.write_log('att_align_{}_{}'.format(_name, i),feat_to_fig(att_align[i,0,:,:].cpu().detach()))
                    self.write_log('att_text_{}_{}'.format(_name, i),self.tokenizer.decode(att_output[i].argmax(dim=-1).tolist()))
                if ctc_output is not None:
                    self.write_log('ctc_text_{}_{}'.format(_name, i),self.tokenizer.decode(ctc_output[i].argmax(dim=-1).tolist(),
                                                                                                    ignore_repeat=True))

    def log_records(self, _name):
        assert hasattr(self, 'dev_wer')
        assert hasattr(self, 'dev_cer')
        assert hasattr(self, 'dev_er')

        dev_wer = self.dev_wer
        dev_cer = self.dev_cer
        dev_er = self.dev_er

        # Ckpt if performance improves
        tasks = []
        if len(dev_er['att']) > 0:
            tasks.append('att')
        if len(dev_er['ctc']) > 0:
            tasks.append('ctc')

        for task in tasks:
            dev_er[task] = sum(dev_er[task])/len(dev_er[task])
            dev_wer[task] = sum(dev_wer[task])/len(dev_wer[task])
            dev_cer[task] = sum(dev_cer[task])/len(dev_cer[task])
            if dev_er[task] < self.best_wer[task][_name]:
                self.best_wer[task][_name] = dev_er[task]
                self.save_checkpoint('best_{}_{}.pth'.format(task, _name), 
                                    self.val_mode,dev_er[task],_name)
            if self.step >= self.max_step:
                self.save_checkpoint('last_{}_{}.pth'.format(task, _name), 
                                    self.val_mode,dev_er[task],_name)
            self.write_log(self.WER,{'dv_'+task+'_'+_name.lower():dev_wer[task]})
            self.write_log(   'cer',{'dv_'+task+'_'+_name.lower():dev_cer[task]})
        
        delattr(self, 'dev_wer')
        delattr(self, 'dev_cer')
        delattr(self, 'dev_er')

    def validate(self, _dv_set, _name):
        # Eval mode
        self.model.eval()

        batch_num = len(_dv_set)
        for batch_id,data in enumerate(_dv_set):
            self.progress('Valid step - {}/{}'.format(batch_id+1,batch_num))
            _, feat, feat_len, txt = data
            feat = feat.to(self.device)
            batch_size = len(feat)

            self.forward_validate(feat, feat_len, txt, self.step, batch_num, batch_id, _name)

        self.log_records(_name)

        # Resume training
        self.model.train()
