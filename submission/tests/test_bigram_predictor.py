"""Tiny delivery-predictor tests; no benchmark data or checkpoint is read."""

import math
import unittest

import torch

from bigram_models import build_model
from research_models import RoPEGPT


CONFIG = {'vocab': 16, 'width': 8, 'heads': 2, 'depth': 1, 'context': 8,
          'ffn': 'swiglu', 'ffn_hidden': 16, 'pos': 'rope', 'dropout': 0.,
          'temperature': 1.3, 'bigram_alpha': 0.25}


class BigramPredictorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def plain_copy(self, model):
        """The same weights without the bigram branch, for exact comparisons."""
        plain = RoPEGPT(dict(CONFIG))
        plain.load_state_dict({name: value for name, value in model.state_dict().items()
                               if name != 'bigram_table'})
        return plain

    def test_output_is_a_normalized_distribution(self):
        model = build_model(CONFIG)
        ids = torch.randint(0, CONFIG['vocab'], (2, CONFIG['context']))
        logp = model.predict_log_probs(ids)
        self.assertEqual(tuple(logp.shape), (2, CONFIG['context'], CONFIG['vocab']))
        self.assertTrue(bool(torch.isfinite(logp).all()))
        torch.testing.assert_close(torch.logsumexp(logp, dim=-1), torch.zeros(2, CONFIG['context']),
                                   atol=1e-5, rtol=0)

    def test_alpha_zero_reproduces_the_plain_student(self):
        model = build_model({**CONFIG, 'bigram_alpha': 0.})
        plain = self.plain_copy(model)
        ids = torch.randint(0, CONFIG['vocab'], (2, CONFIG['context']))
        torch.testing.assert_close(model.predict_log_probs(ids), plain.predict_log_probs(ids),
                                   atol=0, rtol=0)

    def test_mixture_matches_the_preregistered_formula(self):
        model = build_model(CONFIG)
        table = torch.rand(CONFIG['vocab'], CONFIG['vocab'], dtype=torch.float64)
        table = (table / table.sum(-1, keepdim=True)).float()
        with torch.no_grad():
            model.bigram_table.copy_(table)
        plain = self.plain_copy(model)
        ids = torch.randint(0, CONFIG['vocab'], (1, CONFIG['context']))
        with torch.no_grad():
            logp = plain.predict_log_probs(ids)
            expected = torch.logaddexp(logp + math.log1p(-CONFIG['bigram_alpha']),
                                       table[ids].log() + math.log(CONFIG['bigram_alpha']))
        torch.testing.assert_close(model.predict_log_probs(ids), expected, atol=1e-6, rtol=0)

    def test_prediction_at_a_position_ignores_later_tokens(self):
        model = build_model(CONFIG)
        ids = torch.randint(0, CONFIG['vocab'], (1, CONFIG['context']))
        changed = ids.clone()
        changed[0, 4:] = (changed[0, 4:] + 1) % CONFIG['vocab']
        with torch.no_grad():
            first = model.predict_log_probs(ids)
            second = model.predict_log_probs(changed)
        torch.testing.assert_close(first[:, :4], second[:, :4], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
