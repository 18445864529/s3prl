import os
import math
import torch
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MelSpectrogram


class UpstreamExpert(nn.Module):
    """
    Pre-trained weights should be loaded
    """

    def __init__(self, *args, **kwargs):
        super(UpstreamExpert, self).__init__()
        self.output_dim = 160

    # Interface
    def get_output_dim(self):
        return self.output_dim

    # Interface
    def forward(self, wavs):
        """
        Args:
            wavs:
                list of unpadded wavs [wav1, wav2, ...]
                each wav is in torch.FloatTensor with sample rate 16000
                and already put in the device assigned by command-line args

        Return:
            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args
        """
        return wavs
