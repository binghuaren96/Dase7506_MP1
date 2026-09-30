# DASE7506 MP1: frozen small language model

This is the compact, directly scoreable release of the final model. The separate
checkpoint bundle is required; **evaluation does not retrain the model**. The
unmodified course evaluator scores independent 256-token causal windows of the
supplied WikiText-2 text with the fixed train-fitted BPE-2048 tokenizer. Model
selection used validation only. The full-test split was scored after the method
was frozen, solely to report the submitted score.

| Frozen result | Value |
|---|---:|
| Full-test BPB, CPU FP32 | **1.4850591241750035** |
| Validation BPB, CPU FP32 | 1.4647973939045038 |
| Checkpoint SHA-256 | `2145fdb0acf4e07c678335fd6f8a88edb39ede72dbe5039c804a85a7e0ea4169` |
| Checkpoint size | 40,067,992 bytes (38.2077 MiB) |
| Inference assets, including checkpoint | 40,203,767 bytes (38.337 MiB; limit 64 MiB) |
| Paired CPU time / E0 | 3.8861× validation; 3.6931× test (limit 5×) |
| Peak working set | 1.8706 GiB validation; 1.871 GiB test (limit 4 GiB) |

The checkpoint has 5,290,048 trained neural-network parameters and embeds a
2048×2048 float32 bigram table (16 MiB). Its `implementation` field is
`bigram_models`; `bigram_models.py` imports `research_models.py`, which imports
`student.py` and `model.py`. The table is stored inside the checkpoint, not
loaded from an external path at inference. Check the downloaded file before
scoring: `Get-FileHash -Algorithm SHA256 checkpoint.pt` in PowerShell, or
`sha256sum checkpoint.pt` on Linux/macOS. The score belongs to this exact
checkpoint and code version.

## Install and score the frozen checkpoint

Use Python 3.12 from this directory. Create an environment and install one
PyTorch build. For CPU scoring on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe evaluate.py --checkpoint C:\path\to\checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output runs\reproduced-test.json
```

On Linux/macOS, replace the environment and Python paths with
`python3.12 -m venv .venv` and `.venv/bin/python`. On Linux CPU, use the same
PyTorch CPU index command. On macOS, install `torch==2.7.1` from PyPI instead.
The final two commands are otherwise identical with `/` paths. The score is
the `bpb` field in `runs/reproduced-test.json`; do not report `token_ppl` or a
validation score as the full-test score. The output path may be anywhere
writable. The evaluator also saves per-window losses beside its JSON output.
All five supplied benchmark files are checked against `data/manifest.json`
before scoring. No network access or external data are used by evaluation.

## How the model was made

The student was trained **from random initialization**, using only the supplied
train text for gradient updates. It is a six-block, width-256, eight-head
RoPE/SwiGLU language model with residual dropout 0.3. The final training arm
used seed 17, 40,000 updates, batch size 32, 256 targets per example,
AdamW (`β₁=0.9`, `β₂=0.999`, grouped weight decay 0.1/0), a 100-step warmup,
cosine learning-rate schedule with peak 0.0014 and 10% floor, and FP32 GPU
training. That arm processed **327,680,000 target positions**. The original
run took 9,030.5 seconds of process wall time (about 2.51 hours), including
validation diagnostics, on the development GPU; other hardware will differ.

To repeat this recipe, install a suitable CUDA build of `torch==2.7.1` in place
of the CPU wheel, then run in this directory. In the following training and
freezing commands, `python` means this environment's interpreter
(`.\\.venv\\Scripts\\python.exe` on Windows or `.venv/bin/python` on Linux/macOS):

```text
python train_research.py --candidate CAP-p03 --seed 17 --steps 40000 --batch-size 32 --eval-every 500 --peak-lr 0.0014 --adam-beta2 0.999 --device cuda --threads 4 --save-resume-state --run-dir runs/retrain
```

The run writes a separate `checkpoint-step-NNNNNN.pt` at every 500-step
diagnostic. The final student weights are the **equal-weight average of 33
checkpoints, steps 24,000 through 40,000 inclusive at intervals of 500**.
For example, in PowerShell:

```powershell
$parts = 24000..40000 | Where-Object { $_ % 500 -eq 0 } | ForEach-Object { 'runs/retrain/checkpoint-step-{0:D6}.pt' -f $_ }
.\.venv\Scripts\python.exe average_checkpoints.py --output runs\average\checkpoint.pt $parts
```

For bash, the equivalent argument list is:

```bash
.venv/bin/python average_checkpoints.py --output runs/average/checkpoint.pt $(printf 'runs/retrain/checkpoint-step-%06d.pt ' $(seq 24000 500 40000))
```

The frozen predictor applies temperature **T=1.19** to the averaged student's
logits and mixes its distribution with weight **α=0.02** from a train-only
bigram table. `probe_bigram.py` constructs that table from the fixed tokenizer
and train text: a train unigram prior with add-one smoothing, then a fixed
bigram smoothing strength **β=32**. The table is counted once and embedded by
`freeze_final.py`. Each output position reads only the current window's
observed input prefix; no validation or test target enters the table or a
prediction, and no state crosses windows. The original tokenizer/build step
took about 9.8 seconds. The later validation-only postprocessing round used
about 207 seconds for temperature/window/copy probes, 96 seconds for the
five-α bigram probe, and 271 seconds for formal paired validation/resource
checks. These are additional search/evaluation costs, not training targets.

`freeze_final.py` also requires an E0 baseline checkpoint for the 5× paired
timing audit. The original E0 anchor was a seed-17, 4,000-step run with the
supplied GELU model and the same 500-step diagnostic interval; it can be
recreated with:

```text
python train_round1.py --experiment E0 --seed 17 --steps 4000 --batch-size 32 --eval-every 500 --device cuda --threads 4 --run-dir runs/e0
```

Then freeze to a **new, empty** directory and run three alternating CPU FP32
validation checks:

```text
python freeze_final.py --base runs/average/checkpoint.pt --output runs/rebuilt-final --temperature 1.19 --bigram-alpha 0.02 --repeats 3 --baseline runs/e0/checkpoint.pt
```

This is a reproducible *procedure*, not a promise that another GPU, PyTorch
build, operating system, or absolute output path will yield a bit-identical
40,067,992-byte checkpoint. Training numerics, timing and freeze provenance
can change; use the distributed frozen checkpoint for score verification.
The final choice of width, dropout, learning rate, averaging window,
temperature and mixture weight came from validation searches. Only seed 17
was tested for this final recipe, so the small validation improvements carry
selection and seed uncertainty. The report in `../doc/FINAL_REPORT.pdf`
discloses controls, ablations, earlier arms and total search cost.

## Files, provenance and attribution

`common.py`, `evaluate.py`, `data/*`, `model.py`, `train.py`, and
`configs/baseline.json` are copied byte-for-byte from the working course
project. `student.py`, `research_models.py`, `bigram_models.py`, the training,
averaging and freezing scripts, and focused tests contain the student work.
The checkpoint is distributed separately and is intentionally absent from this
code directory. Local experiment outputs under `runs/` are not needed to score
the frozen predictor.

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury
and Richard Socher in *Pointer Sentinel Mixture Models*. The text comes from
Wikipedia contributors; the upstream dataset lists CC BY-SA 3.0 and the GNU
Free Documentation License. The supplied raw-v1 revision is
`b08601e04326c79dfdd32d625aee71d232d685c3`.

Claude Code and Codex provided substantive assistance with experiment
planning, code implementation, testing, auditing and documentation. The
student remains responsible for the method, results and submission.
