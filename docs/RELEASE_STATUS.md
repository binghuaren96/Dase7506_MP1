# Release status

The model and method are frozen. Checkpoint SHA-256 `2145fdb0acf4e07c678335fd6f8a88edb39ede72dbe5039c804a85a7e0ea4169`; official CPU FP32 validation BPB `1.4647973939045038`; frozen full-test BPB `1.4850591241750035`. No further test-based selection is planned.

The released tree was checked before submission. The 26 copied sources in `submission/` match their originals byte for byte, a fresh evaluation from `submission/` against the separate bundle checkpoint reproduced both BPB values exactly, and the report renders to four pages. The verification records are in `evidence/`.
