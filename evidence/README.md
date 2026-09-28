# Selected evidence for the frozen submission

This directory contains a small, inspectable selection from the local experiment archive. The full 21.6 GiB `code/runs/` tree is preserved locally but is not part of the code submission. The selected reports are condensed from their originals in `code/`; the selected JSONs record the frozen run's outputs with machine-specific path prefixes removed. File names were shortened only to make the release easy to navigate.

| Release file | Local source |
|---|---|
| `round1-findings.md` | `code/ROUND1_FINDINGS.md` |
| `round1-extension-findings.md` | `code/ROUND1_EXTENSION_FINDINGS.md` |
| `dropout-findings.md` | `code/P4_FINDINGS.md` |
| `posthoc-findings.md` | `code/POSTHOC_FINDINGS_20260928.md` |
| `experiment-ledger.md` | `code/EXPERIMENT_LEDGER.md` |
| `bigram-probe.json` | `code/runs/posthoc-20260928/bigram/L40hi-selected.json` |
| `final-validation.json` | `code/runs/posthoc-20260928/final/frozen-T1.19/validation_cpu_fp32.json` |
| `final-resource-check.json` | `code/runs/posthoc-20260928/final/resource-check.json` |
| `final-test-scoring.json` | `code/runs/posthoc-20260928/final/test-scoring/test-scoring.json` |
| `release-validation.json` | Fresh evaluation from `submission/` with the separate bundle checkpoint |
| `release-test.json` | Fresh evaluation from `submission/` with the separate bundle checkpoint |

The course evaluator records SHA-256 values for the checkpoint, implementation, evaluator and tokenizer in its score JSON. `final-resource-check.json` records the matching checkpoint SHA and the three resource limits. The two `release-*.json` results reproduce the original validation and full-test BPB exactly from the compact submission directory. A reviewer should still hash and evaluate the downloaded checkpoint independently.
