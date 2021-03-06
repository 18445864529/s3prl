import os
import torch

from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert


def vq_wav2vec_local(ckpt, feature_selection=None, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
            feature_selection (str): 'z' (default) or 'codewords'
    """
    assert os.path.isfile(ckpt)
    if feature_selection not in ['z', 'codewords']:
        feature_selection = 'z'
    return _UpstreamExpert(ckpt, feature_selection, *args, **kwargs)


def vq_wav2vec_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): URL
            feature_selection (str): 'z' or 'codewords'
            refresh (bool): whether to download ckpt/config again if existed
    """
    return vq_wav2vec_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def vq_wav2vec(refresh=False, *args, **kwargs):
    """
        The default model - Large model with context vector
            feature_selection (str): 'z' or 'codewords'
            refresh (bool): whether to download ckpt/config again if existed
    """
    return vq_wav2vec_gumbel(refresh=refresh, *args, **kwargs)


def vq_wav2vec_gumbel(refresh=False, *args, **kwargs):
    """
        The Gumbel model
            feature_selection (str): 'z' or 'codewords'
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt'
    return vq_wav2vec_url(refresh=refresh, *args, **kwargs)


def vq_wav2vec_kmeans(refresh=False, *args, **kwargs):
    """
        The K-means model
            feature_selection (str): 'z' or 'codewords'
            refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs['ckpt'] = 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt'
    return vq_wav2vec_url(refresh=refresh, *args, **kwargs)
