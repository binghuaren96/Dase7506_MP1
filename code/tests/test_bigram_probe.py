"""Tiny train-built bigram tests; no benchmark data or checkpoint is read."""

import math
import unittest

import torch
from torch import nn

from evaluate import score
from probe_bigram import bigram_table, score_candidates


class UniformModel(nn.Module):
    def predict_log_probs(self, ids):
        return torch.full((*ids.shape, 2048), -math.log(2048), dtype=torch.float32)


class BigramProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_smoothing_normalizes_and_keeps_the_unigram_prior(self):
        tokens = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        table, metadata = bigram_table(tokens, beta=1.0, vocab=4)
        torch.testing.assert_close(table.sum(-1), torch.ones(4), atol=1e-6, rtol=0)
        self.assertTrue(bool((table > 0).all()))
        self.assertEqual(metadata['train_tokens'], 8)
        self.assertEqual(metadata['table_bytes'], 4 * 4 * 4)
        self.assertEqual(metadata['observed_pairs'], 4)
        # Row 0 observes two (0,1) transitions out of two: two thirds observed mass,
        # one third spread over the uniform train unigram.
        torch.testing.assert_close(table[0], torch.tensor([0.25, 2.25, 0.25, 0.25]) / 3,
                                   atol=1e-6, rtol=0)

    def test_alpha_zero_matches_evaluate_and_ignores_the_table(self):
        tokens = torch.tensor([1, 2, 1, 3, 1, 4, 1, 5], dtype=torch.long)
        model = UniformModel()
        flat = torch.full((2048, 2048), 1 / 2048)
        skewed = torch.full((2048, 2048), 0.4 / 2047)
        skewed[:, 0] = 0.6
        fixed = score(model, tokens, 17, torch.device('cpu'), 'fp32')
        for table in (flat, skewed):
            probed = score_candidates(model, table, tokens, byte_count=17, alphas=(0.0,))
            self.assertEqual(probed['rows']['0']['nll_nats'], fixed['nll_nats'])
            self.assertEqual(probed['rows']['0']['bpb'], fixed['bpb'])

    def test_a_flat_bigram_leaves_the_mixture_unchanged(self):
        tokens = torch.tensor([1, 2, 1, 3, 1, 4, 1, 5], dtype=torch.long)
        model = UniformModel()
        table = torch.full((2048, 2048), 1 / 2048)
        probed = score_candidates(model, table, tokens, byte_count=17, alphas=(0.0, 0.5))
        # p_bigram equals the uniform student, so any mixture is the same distribution.
        self.assertLess(abs(probed['rows']['0.5']['delta_bpb_vs_alpha0']), 1e-5)


if __name__ == '__main__':
    unittest.main()
