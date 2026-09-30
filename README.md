# DASE7506 MP1 — Small Language Model Challenge

Student ID:3036781636  Name:Ren Binghua

The frozen method is a 5.29M-parameter RoPE/SwiGLU student trained from random initialization on the supplied WikiText-2 train text. The frozen predictor averages 33 checkpoints from steps 24,000–40,000, applies temperature 1.19, and mixes in a train-only smoothed bigram table with weight 0.02. Development and model selection used validation; the full-test split was scored only after the predictor was frozen.

| Frozen result | Value |
|---|---:|
| Official CPU FP32 validation BPB | **1.4647973939045038** |
| Official CPU FP32 full-test BPB | **1.4850591241750035** |
| Paired CPU scoring time | 3.8861× E0 on validation; 3.6931× on test (limit 5×) |
| Peak evaluation RAM | 1.871 GiB on test (limit 4 GiB) |
| All uncompressed inference assets | 38.337 MiB (limit 64 MiB) |
| Checkpoint SHA-256 | `2145fdb0acf4e07c678335fd6f8a88edb39ede72dbe5039c804a85a7e0ea4169` |

## Repository map

| Path | Purpose |
|---|---|
| [`submission/`](submission/) | Minimal runnable code, fixed data/tokenizer, installation and exact evaluation instructions. |
| [`docs/FINAL_REPORT.pdf`](docs/FINAL_REPORT.pdf) | Technical report (at most 10 pages). The Markdown source is alongside it. |
| [`evidence/`](evidence/) | Selected experiment records and official score/resource JSONs underlying the report. |
| [`GUIDE.md`](GUIDE.md) | Supplied assignment rules. |
| [`docs/RELEASE_STATUS.md`](docs/RELEASE_STATUS.md) | Frozen scores and the local verification record. |

The full 21.6 GiB experiment history remains in the local `code/runs/` archive and is excluded from the public code repository. The original working `code/` tree is preserved locally; `submission/` is the clean evaluation entry point. The final checkpoint is distributed as a separate versioned bundle, identified by the SHA-256 above. A code-only checkout does not include model weights.

## Evaluate the frozen checkpoint

Follow [`submission/README.md`](submission/README.md) to install dependencies. From `submission/`, with the downloaded checkpoint at any path, run:

```bash
python evaluate.py --checkpoint /path/to/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The scorer, tokenizer, data files and independent 256-token causal-window protocol are the supplied course versions. The bigram table is embedded in the checkpoint; no external dataset or model download is needed to evaluate it.

Substantive implementation, experimentation, analysis and documentation were assisted by Claude Code and OpenAI Codex. The student remains responsible for understanding, reviewing and submitting the work.
