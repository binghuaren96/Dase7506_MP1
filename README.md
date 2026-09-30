# DASE7506 MP1 — Small Language Model Challenge

Student ID: 3036781636

Name: Ren Binghua

The final predictor is a 5.29M-parameter RoPE/SwiGLU language model trained from random initialization on the supplied WikiText-2 training split. It averages 33 late checkpoints, applies temperature 1.19, and mixes in a train-only bigram table with weight 0.02. Model selection used validation; the full-test split was scored after the predictor was frozen.

| Frozen CPU FP32 result | Value |
|---|---:|
| Validation BPB | 1.4647973939045038 |
| Full-test BPB | **1.4850591241750035** |
| Paired scoring time vs E0 | 3.8861× validation; 3.6931× test (limit 5×) |
| Peak evaluation RAM | 1.871 GiB on test (limit 4 GiB) |
| Uncompressed inference assets | 38.337 MiB (limit 64 MiB) |

## Submission files

- [`code/`](code/) contains the unchanged course evaluator, fixed data and tokenizer, model code, tests, and exact installation, training, and scoring instructions in [`code/README.md`](code/README.md).
- [`doc/FINAL_REPORT.pdf`](doc/FINAL_REPORT.pdf) is the technical report. It includes the baseline, equal-target comparison, ablation, search cost, and limitations.

The matching checkpoint is distributed separately from this code repository. Its SHA-256 is `2145fdb0acf4e07c678335fd6f8a88edb39ede72dbe5039c804a85a7e0ea4169` (40,067,992 bytes). Download it from the checkpoint link in the course submission, verify its hash, then run from `code/`:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

Evaluation requires the matching checkpoint but no retraining, external data, or network access. The model's bigram table is embedded in the checkpoint.

Claude Code and OpenAI Codex provided substantive assistance with planning, implementation, testing, auditing, and documentation. The student remains responsible for understanding and submitting the work.
