# MP1 frozen checkpoint bundle

This bundle contains the frozen predictor selected on the validation split. Download the matching immutable code release and follow its `submission/README.md`; no training is needed. The `checkpoint.pt` file embeds the train-only bigram table, so the scorer needs no auxiliary model file.

| Item | Value |
|---|---|
| Checkpoint | `checkpoint.pt` |
| SHA-256 | `2145fdb0acf4e07c678335fd6f8a88edb39ede72dbe5039c804a85a7e0ea4169` |
| Bytes | 40,067,992 |
| Model module | `bigram_models` |
| Official CPU FP32 full-test BPB | `1.4850591241750035` |
| Official CPU FP32 validation BPB | `1.4647973939045038` |

Extract the archive, verify its SHA-256 checksum, then from the release's `submission/` directory run:

```bash
python evaluate.py --checkpoint /path/to/extracted/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The code link and this bundle refer to the same release. The checkpoint was frozen before the full-test score was measured, so the public test text is not used to tune any model.
