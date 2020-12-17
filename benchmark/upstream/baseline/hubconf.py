import os

from .expert import UpstreamExpert as _UpstreamExpert


def baseline(config, *args, **kwargs):
    """
        Baseline feature
            config: PATH
    """
    assert os.path.isfile(config)
    return _UpstreamExpert(config, *args, **kwargs)


def baseline_default(*args, **kwargs):
    """
        Default baseline feature - Fbank, or Mel-scale spectrogram
    """
    return baseline_fbank(*args, **kwargs)


def baseline_spectrogram(*args, **kwargs):
    """
        Baseline feature - Linear-scale spectrogram
    """
    kwargs['config'] = 'benchmark/upstream/baseline/spectrogram.yaml'
    return baseline(*args, **kwargs)


def baseline_fbank(*args, **kwargs):
    """
        Baseline feature - Fbank, or Mel-scale spectrogram
    """
    kwargs['config'] = 'benchmark/upstream/baseline/fbank.yaml'
    return baseline(*args, **kwargs)


def baseline_mfcc(*args, **kwargs):
    """
        Baseline feature - MFCC
    """
    kwargs['config'] = 'benchmark/upstream/baseline/mfcc.yaml'
    return baseline(*args, **kwargs)