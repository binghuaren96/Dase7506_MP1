"""Delivery predictor: a frozen RoPE student mixed with a train-only bigram table.

The table is counted from the course train split with the fixed tokenizer and smoothed
by the train unigram prior at the fixed strength ``beta`` (see ``probe_bigram``). At
every position the delivery distribution is

    p = (1 - alpha) * P_student(. | x[0:t+1]) + alpha * p_bigram(. | x_t)

Both components read observed inputs only, so each 256-token window stays independent
and strictly causal and no validation or test target ever enters the prediction. The
student's ``temperature`` calibration is applied before the mixture, exactly as in the
probes that chose these values.

``build_model`` follows the same contract as ``research_models.build_model``, so the
unmodified course ``evaluate.py`` scores this predictor. The table travels inside the
checkpoint as a float32 buffer, so the delivery bundle is a single file and scoring
needs no external path.
"""

import math

import torch

from research_models import RoPEGPT


class BigramRoPEGPT(RoPEGPT):
    """RoPE student whose evaluation distribution is mixed with a bigram table."""

    def __init__(self, config):
        super().__init__(config)
        alpha = config.get('bigram_alpha')
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError('config must carry a numeric bigram_alpha.')
        if not 0. <= alpha < 1.:
            raise ValueError('bigram_alpha must lie in [0, 1).')
        self.alpha = float(alpha)
        # A uniform table keeps the transform a valid normalized distribution even
        # before a state dict is loaded; the frozen checkpoint always overwrites it.
        self.register_buffer('bigram_table',
                             torch.full((config['vocab'], config['vocab']),
                                        1. / config['vocab']))

    def predict_log_probs(self, ids):
        logp = super().predict_log_probs(ids)
        if self.alpha == 0.:
            return logp
        # Row x_t holds p_bigram(. | x_t): the mixture conditions only on the observed
        # input token, never on the position being predicted.
        rows = self.bigram_table[ids]
        return torch.logaddexp(logp + math.log1p(-self.alpha),
                               rows.log() + math.log(self.alpha))


def build_model(config):
    if config.get('pos') != 'rope':
        raise ValueError('The bigram predictor is defined on the RoPE student only.')
    return BigramRoPEGPT(config)
