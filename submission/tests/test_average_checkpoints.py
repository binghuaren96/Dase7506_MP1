"""Small file-level tests for safe checkpoint averaging."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from average_checkpoints import average_checkpoints, file_sha256
from common import PROTOCOL, make_model


CONFIG = dict(vocab=2048, width=32, heads=4, depth=1, context=256,
              ffn="swiglu", ffn_hidden=88, pos="rope", dropout=0.2)


class CheckpointAveragingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        model, _ = make_model("research_models", CONFIG, torch.device("cpu"))
        cls.reference = {
            "protocol": PROTOCOL,
            "implementation": "research_models",
            "config": CONFIG,
            "model": model.state_dict(),
            "seed": 17,
            "train_tokens": 10 * 32 * 256,
        }

    def test_identical_inputs_preserve_weights_and_evaluator_fields(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / "run_a" / "checkpoint.pt"
            second = root / "run_b" / "checkpoint-step-000020.pt"
            first.parent.mkdir()
            second.parent.mkdir()
            torch.save(self.reference, first)
            (first.parent / "metrics.json").write_text(json.dumps({
                "checkpoint_sha256": file_sha256(first), "steps": 10,
                **{field: self.reference[field] for field in
                   ("protocol", "implementation", "config", "seed", "train_tokens")},
            }), encoding="utf-8")
            later = copy.deepcopy(self.reference)
            later["train_tokens"] = 20 * 32 * 256
            torch.save(later, second)
            output = root / "average" / "checkpoint.pt"

            summary = average_checkpoints([first, second], output)
            result = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(result["protocol"], PROTOCOL)
            self.assertEqual(result["implementation"], "research_models")
            self.assertEqual(result["config"], CONFIG)
            self.assertEqual(result["seed"], 17)
            self.assertEqual(result["train_tokens"], 20 * 32 * 256)
            self.assertEqual(result["steps"], 20)
            self.assertEqual([entry["step"] for entry in result["averaging"]["sources"]], [10, 20])
            self.assertEqual([entry["step_source"] for entry in result["averaging"]["sources"]],
                             ["verified_metrics", "filename"])
            self.assertEqual([entry["sha256"] for entry in result["averaging"]["sources"]],
                             [file_sha256(first), file_sha256(second)])
            self.assertEqual(summary["sha256"], file_sha256(output))
            for name, expected in self.reference["model"].items():
                torch.testing.assert_close(result["model"][name], expected, rtol=0, atol=0)
            model, _ = make_model(result["implementation"], result["config"], torch.device("cpu"))
            model.load_state_dict(result["model"])
            self.assertIs(model.head.weight, model.token.weight)
            with self.assertRaises(FileExistsError):
                average_checkpoints([first, second], output)

    def test_rejects_incompatible_or_nonfloating_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first = root / "first" / "checkpoint-step-000010.pt"
            first.parent.mkdir()
            torch.save(self.reference, first)

            def missing_key(checkpoint):
                checkpoint["model"].pop("norm.bias")

            def wrong_shape(checkpoint):
                checkpoint["model"]["norm.weight"] = checkpoint["model"]["norm.weight"][:-1]

            def wrong_dtype(checkpoint):
                checkpoint["model"]["norm.weight"] = checkpoint["model"]["norm.weight"].double()

            def wrong_config(checkpoint):
                checkpoint["config"]["dropout"] = 0.1

            def wrong_seed(checkpoint):
                checkpoint["seed"] = 29

            def nonfloating(checkpoint):
                checkpoint["model"]["norm.weight"] = checkpoint["model"]["norm.weight"].long()

            cases = [
                ("keys", missing_key, "state keys"),
                ("shape", wrong_shape, "shape"),
                ("dtype", wrong_dtype, "dtype"),
                ("config", wrong_config, "config"),
                ("seed", wrong_seed, "seed"),
                ("nonfloating", nonfloating, "Non-floating"),
            ]
            for label, mutate, message in cases:
                with self.subTest(label=label):
                    second = root / label / "checkpoint-step-000020.pt"
                    second.parent.mkdir()
                    altered = copy.deepcopy(self.reference)
                    altered["train_tokens"] = 20 * 32 * 256
                    mutate(altered)
                    torch.save(altered, second)
                    output = root / label / "output.pt"
                    with self.assertRaisesRegex(ValueError, message):
                        average_checkpoints([first, second], output)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
