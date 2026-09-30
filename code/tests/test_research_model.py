"""Contract checks for the round-two RoPE variant, the round-one compatibility path and the LR floor."""
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import torch

import test_contract
from common import PROTOCOL, ROOT, make_model
from research_models import RoPEGPT, apply_rope, build_model, rope_tables
from student import SwiGLUGPT
from student import build_model as build_round_one
from train_research import CANDIDATES, LR_FLOORS, learning_rate

P3A_CONFIG = dict(vocab=2048, width=192, heads=6, depth=4, context=256,
                  ffn='swiglu', ffn_hidden=512, pos='rope')

ANCHOR = ROOT / 'runs/round1-extension-4000-20260926/E3-s17/checkpoint.pt'
ANCHOR_LR_SHA256 = 'e9c3d46bc8704f230259efda35102734b841a01107d5ee14d594ae28566778b1'


class RoPEContractTests(test_contract.ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                                      ffn='swiglu', ffn_hidden=88, pos='rope')).eval()


class RoPEModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = dict(vocab=2048, width=128, heads=4, depth=4, context=256,
                           ffn='swiglu', ffn_hidden=344)

    def test_rope_parameter_budget_drops_the_position_table(self):
        model = build_model(self.config | {'pos': 'rope'})
        self.assertIsInstance(model, RoPEGPT)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_060_288)
        self.assertNotIn('pos.weight', model.state_dict())
        self.assertIs(model.head.weight, model.token.weight)

    def test_learned_position_configuration_is_identical_to_round_one(self):
        torch.manual_seed(29)
        reference = build_round_one(dict(self.config)).eval()
        torch.manual_seed(29)
        delegated = build_model(self.config | {'pos': 'learned'}).eval()
        self.assertIsInstance(delegated, SwiGLUGPT)
        self.assertEqual(set(reference.state_dict()), set(delegated.state_dict()))
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, delegated.state_dict()[name], rtol=0, atol=0)
        ids = torch.randint(0, 2048, (2, 9))
        with torch.no_grad():
            torch.testing.assert_close(reference(ids), delegated(ids), rtol=0, atol=0)

    def test_rotation_preserves_norms_and_acts_only_by_position(self):
        cos, sin = rope_tables(7, 8, torch.device('cpu'))
        x = torch.randn(3, 3, 7, 8)
        rotated = apply_rope(x, cos, sin)
        torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
        # Position zero carries no rotation; later positions do.
        torch.testing.assert_close(rotated[..., 0, :], x[..., 0, :], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(rotated[..., 3, :], x[..., 3, :], atol=1e-4))

    def test_invalid_position_settings_fail_explicitly(self):
        for updates in ({'pos': 'sinusoidal'}, {'pos': 'rope', 'ffn': 'gelu'},
                        {'pos': 'rope', 'width': 33, 'heads': 33}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                build_model(self.config | updates)


class CandidateTests(unittest.TestCase):
    def test_candidate_parameter_counts_match_the_plan(self):
        expected = {'C-depth': 1_492_000, 'C-width': 2_223_232, 'R-rope': 1_060_288,
                    'T-lrfloor': 1_093_056, 'P3A': 2_174_080,
                    'CAP-4k': 5_290_048, 'CAP-30k-p0': 5_290_048, 'CAP-30k-p01': 5_290_048,
                    'P3A-30k': 2_174_080, 'CAP-p0': 5_290_048,
                    'CAP-p02': 5_290_048, 'CAP-p03': 5_290_048}
        self.assertEqual(set(CANDIDATES), set(expected))
        for name, spec in CANDIDATES.items():
            with self.subTest(candidate=name):
                model, _ = make_model(spec['implementation'], spec['config'], torch.device('cpu'))
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected[name])


class P3ACombinationContractTests(test_contract.ContractTests):
    """The combination arm must satisfy the same contract as every frozen model."""

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = build_model(P3A_CONFIG).eval()


class P3ACombinationTests(unittest.TestCase):
    def test_parameter_budget_drops_exactly_the_position_table(self):
        model = build_model(P3A_CONFIG)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2_174_080)
        self.assertEqual(2_223_232 - 256*192, 2_174_080, 'C-width minus its 256x192 position table.')
        self.assertNotIn('pos.weight', model.state_dict())
        self.assertEqual(model.context, 256)
        self.assertIs(model.head.weight, model.token.weight)

    def test_cpu_checkpoint_round_trip_preserves_predictions(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        model = build_model(P3A_CONFIG).eval()
        ids = torch.randint(0, 2048, (2, 17), generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            expected = model.predict_log_probs(ids)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            torch.save({'protocol': PROTOCOL, 'implementation': 'research_models', 'seed': 17,
                        'config': dict(model.config), 'train_tokens': 0,
                        'model': {name: value.detach().cpu() for name, value in model.state_dict().items()}},
                       path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            reloaded, _ = make_model('research_models', checkpoint['config'], torch.device('cpu'))
            reloaded.load_state_dict(checkpoint['model'])
            with torch.no_grad():
                torch.testing.assert_close(expected, reloaded.eval().predict_log_probs(ids),
                                           rtol=0, atol=0)


class DropoutTests(unittest.TestCase):
    """Residual dropout must be absent by default, inert at p=0, and paired-arm safe."""

    def setUp(self):
        torch.set_num_threads(2)
        self.config = dict(vocab=2048, width=64, heads=4, depth=2, context=256,
                           ffn='swiglu', ffn_hidden=176, pos='rope')

    def build(self, **updates):
        return build_model(self.config | updates)

    def test_paired_arms_start_from_bit_identical_weights(self):
        torch.manual_seed(17)
        frozen = self.build(dropout=0.)
        torch.manual_seed(17)
        dropped = self.build(dropout=.3)
        self.assertEqual(set(frozen.state_dict()), set(dropped.state_dict()))
        for name, value in frozen.state_dict().items():
            torch.testing.assert_close(value, dropped.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in frozen.parameters()),
                         sum(p.numel() for p in dropped.parameters()))

    def test_omitting_dropout_reproduces_the_frozen_configuration(self):
        torch.manual_seed(17)
        default = self.build().eval()
        torch.manual_seed(17)
        explicit = self.build(dropout=0.).eval()
        ids = torch.randint(0, 2048, (2, 9), generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            torch.testing.assert_close(default.predict_log_probs(ids),
                                       explicit.predict_log_probs(ids), rtol=0, atol=0)

    def test_dropout_is_inert_in_eval_and_active_in_train(self):
        model = self.build(dropout=.3)
        ids = torch.randint(0, 2048, (4, 16), generator=torch.Generator().manual_seed(5))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(ids), model(ids), rtol=0, atol=0)
        model.train()
        with torch.no_grad():
            self.assertFalse(torch.allclose(model(ids), model(ids)),
                             'Train-mode dropout must actually perturb the activations.')

    def test_a_frozen_checkpoint_still_loads_strictly_into_a_dropout_model(self):
        torch.manual_seed(17)
        frozen = self.build().eval()
        ids = torch.randint(0, 2048, (2, 9), generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            expected = frozen.predict_log_probs(ids)
        torch.manual_seed(29)
        dropped = self.build(dropout=.2)
        dropped.load_state_dict(frozen.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(expected, dropped.eval().predict_log_probs(ids),
                                       rtol=0, atol=0)

    def test_dropout_does_not_disturb_an_independent_sampling_stream(self):
        """Window sampling draws from a private generator, so p cannot move the hashes."""
        model = self.build(dropout=.3).train()
        ids = torch.randint(0, 2048, (2, 9))
        generator = torch.Generator().manual_seed(7)
        expected = torch.randint(0, 10**6, (8,), generator=generator)
        generator = torch.Generator().manual_seed(7)
        model(ids)
        torch.testing.assert_close(torch.randint(0, 10**6, (8,), generator=generator), expected)

    def test_invalid_dropout_is_rejected(self):
        for dropout in (1., -.1, 2):
            with self.subTest(dropout=dropout), self.assertRaises(ValueError):
                self.build(dropout=dropout)


class LearningRateTests(unittest.TestCase):
    def test_default_floor_reproduces_the_recorded_round_one_sequence(self):
        digest = hashlib.sha256()
        for step in range(4000):
            digest.update(struct.pack('<d', learning_rate(step, 4000, .1)))
        self.assertEqual(digest.hexdigest(), ANCHOR_LR_SHA256)

    def test_raised_floor_only_lifts_the_tail_of_the_schedule(self):
        base = [learning_rate(step, 4000, .1) for step in range(4000)]
        raised = [learning_rate(step, 4000, .2) for step in range(4000)]
        self.assertEqual(set(LR_FLOORS), {.1, .2})
        self.assertEqual(base[0], raised[0], 'Both schedules share the warmed-up peak.')
        self.assertLess(raised[-1], raised[-2], 'The tail still decays.')
        for step in range(100, 4000):
            self.assertGreater(raised[step], base[step])
        self.assertAlmostEqual(raised[-1]/.001, .2, places=4)
        self.assertAlmostEqual(base[-1]/.001, .1, places=3)


class AnchorCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(ANCHOR.exists(), 'The completed E3 seed-17 checkpoint is absent.')
    def test_round_one_checkpoint_loads_through_the_research_builder(self):
        torch.set_num_threads(2)
        checkpoint = torch.load(ANCHOR, map_location='cpu', weights_only=True)
        models = []
        for implementation in ('student', 'research_models'):
            model, _ = make_model(implementation, checkpoint['config'], torch.device('cpu'))
            model.load_state_dict(checkpoint['model'])
            models.append(model.eval())
        ids = torch.randint(0, 2048, (2, 17), generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            torch.testing.assert_close(models[0](ids), models[1](ids), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
