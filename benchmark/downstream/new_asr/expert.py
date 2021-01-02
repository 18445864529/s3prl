import os
import sys
import math
import torch
import random
import argparse
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .bin.train_asr import Solver
from .src.audio import Augment

HALF_BATCHSIZE_AUDIO_LEN = 800


class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream, expdir, data, hparas, model, **kwargs):
        """
        Args:
            upstream_dim: int
                Different upstream will give different representation dimension
                You might want to first project them to the same dimension
            
            downstream_expert: dict
                The 'downstream_expert' field specified in your downstream config file
                eg. benchmark/downstream/example/config.yaml

            **kwargs: dict
                The arguments specified by the argparser in run_benchmark.py
                in case you need it.
        """

        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream = upstream

        config = {
            'data': data,
            'hparas': hparas,
            'model': model,
        }
        paras = self._get_pseudo_args(expdir)

        self.solver = Solver(config, paras, mode='train', feat_dim=upstream_dim)
        if upstream == 'dummy':
            self.solver.load_data(for_s3prl=True)
        else:
            self.solver.load_wav()
        self.solver.set_model()

        self.do_specaug = config['data']['audio']['augment']
        self.do_specaug_in_cpu = config.get('do_specaug_in_cpu', True)
        self.specaug = Augment()

    def _get_pseudo_args(self, expdir):
        # Arguments
        parser = argparse.ArgumentParser(description='Training E2E asr.')
        parser.add_argument('--config', type=str, help='Path to experiment config.')
        parser.add_argument('--name', default=None, type=str, help='Name for logging.')
        parser.add_argument('--logdir', default='log/', type=str, help='Logging path.', required=False)
        parser.add_argument('--ckpdir', default='ckpt/', type=str, help='Checkpoint path.', required=False)
        parser.add_argument('--outdir', default='result/', type=str, help='Decode output path.', required=False)
        parser.add_argument('--load', default=None, type=str, help='Load pre-trained model (for training only)', required=False)
        parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducable results.', required=False)
        parser.add_argument('--cudnn-ctc', action='store_true', help='Switches CTC backend from torch to cudnn')
        parser.add_argument('--njobs', default=4, type=int, help='Number of threads for dataloader/decoding.', required=False)
        parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
        parser.add_argument('--no-pin', action='store_true', help='Disable pin-memory for dataloader')
        parser.add_argument('--test', action='store_true', help='Test the model.')
        parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
        parser.add_argument('--lm', action='store_true', help='Option for training RNNLM.')
        parser.add_argument('--amp', action='store_true', help='Option to enable AMP.')
        parser.add_argument('--reserve_gpu', default=0, type=float, help='Option to reserve GPU ram for training.')
        parser.add_argument('--jit', action='store_true', help='Option for enabling jit in pytorch. (feature in development)')
        parser.add_argument('--cuda', default=0, type=int, help='Choose which gpu to use.')
        parser.add_argument('--deterministic', action='store_true', help='Ensuring same behavior')
        
        paras = parser.parse_args([
            '--deterministic',
            '--name', f'{os.path.basename(expdir)}',
            '--njobs', '16',
            '--seed', '0',
            '--logdir', f'{os.path.dirname(__file__)}/log/',
            '--ckpdir', f'{os.path.dirname(__file__)}/ckpt/',
            '--outdir', f'{os.path.dirname(__file__)}/result/',
            '--reserve_gpu', '0',
        ])
        setattr(paras,'gpu',not paras.cpu)
        setattr(paras,'pin_memory',not paras.no_pin)
        setattr(paras,'verbose',not paras.no_msg)

        # Hack to preserve GPU ram just incase OOM later on server
        if paras.gpu and paras.reserve_gpu>0:
            buff = torch.randn(int(paras.reserve_gpu*1e9//4)).to(torch.device('cuda:' + str(paras.cuda)))
            del buff

        return paras

    """
    Datalaoder Specs:
        Each dataloader should output a list in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with:
            1. dim() == 1
            2. sample_rate == 16000
            3. directly loaded by torchaudio without any preprocessing
    """

    # Interface
    def get_train_dataloader(self):
        return self.solver.tr_set

    # Interface
    def get_dev_dataloader(self):
        return self.solver.dv_set

    # Interface
    def get_test_dataloader(self):
        raise NotImplementedError

    # Interface
    def forward(self, features, text, global_step, prefix, batch_num, batch_id, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            your_other_contents1, ... :
                in the order defined by your dataloader (dataset + collate_fn)
                these are all in cpu, and you can move them to the same device
                as features

            records:
                defaultdict(list), by dumping contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records

                Note1. benchmark/runner.py will call self.log_records
                    1. every log_step during training
                    2. once after evalute the whole dev/test dataloader

                Note2. log_step is defined in your downstream config

            logger:
                Tensorboard SummaryWriter, given here for logging/debugging convenience
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging

        Return:
            loss:
                the loss to be optimized, should not be detached
                a single scalar in torch.FloatTensor
        """
        device = features[0].device
        
        if self.upstream != 'dummy':
            if HALF_BATCHSIZE_AUDIO_LEN < 3500 and self.training:
                max_len = [len(feat) for feat in features][0]
                if max_len > HALF_BATCHSIZE_AUDIO_LEN:
                    features = features[::2]
                    text = text[::2]

            if self.training and self.do_specaug:
                aug_device = device
                if self.do_specaug_in_cpu:
                    aug_device = 'cpu'
                features = [self.specaug.to(aug_device)(feat.to(aug_device)).to(device) for feat in features]
        
        feat_len = torch.LongTensor([len(feat) for feat in features])
        features = pad_sequence(features, batch_first=True)
        text = pad_sequence(text, batch_first=True)

        if self.training:
            loss = self.solver.forward_train(features, feat_len, text, global_step)
        else:
            _name = self.solver.dv_names
            self.solver.forward_validate(features, feat_len, text, batch_num, batch_id, _name)
            loss = torch.zeros(1)

        return loss

    # interface
    def log_records(self, prefix, **kwargs):
        """
        This function will be used in both train/dev/test, you can use
        self.training (bool) to control the different behavior for
        training or evaluation (dev/test)

        Args:
            records:
                defaultdict(list), contents already prepared by self.forward

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        if not self.training:
            _name = self.solver.dv_names
            self.solver.log_records(_name)
