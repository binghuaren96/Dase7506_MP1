# DASE7506 MP1: A Compact Causal Language Model with Train-Only Bigram Calibration

**Technical report - 28 September 2026 - final model selected on validation**

## Abstract

We train a 5.29-million-parameter causal language model from random initialization on the supplied WikiText-2 training text. The model uses SwiGLU feed-forward blocks, rotary position embeddings, residual dropout, a 40,000-update cosine schedule, and an average of 33 late checkpoints. At inference, a fixed temperature and a small bigram distribution estimated solely from the training split are mixed with the model prediction. The frozen predictor obtains **1.46479739 validation BPB** and **1.48505912 full-test BPB** with the unchanged CPU FP32 evaluator. It uses **3.886x** the paired baseline validation scoring time, **1.871 GiB** peak evaluation working set, and **38.337 MiB** of uncompressed inference assets, meeting all three limits. Equal-target baseline comparisons, a paired dropout ablation, post-processing ablations, and the complete formal-run search budget are reported below. The small final validation gain is subject to single-seed and repeated-selection uncertainty.

## 1. Task, metric, and protocol

The supplied benchmark fixes the WikiText-2 text, a BPE tokenizer with vocabulary size 2,048, an unchanged evaluator, and independent causal windows of 256 tokens. We trained weights and estimated the bigram table only from the supplied **train** split. Validation was used to select architecture, training settings, checkpoints, temperature, and mixture strength. No pretrained weights or external text were used. The predictor reads only the prefix available within its current window; it carries no state between windows and needs no network access at evaluation.

For total target negative log-likelihood `L` in nats and `B` original UTF-8 bytes, the evaluator reports `BPB = L / (B * ln(2))`. This is a byte-normalized score, not token perplexity. All scores called *official* below come from the supplied `evaluate.py` on CPU in FP32 with four threads. Training-time GPU diagnostics and CPU probes are labeled separately. The **final candidate** was frozen before its full-test evaluation; it was then scored three times for timing and numerical reproducibility, never to choose settings.

## 2. Model and training recipe

The starting E0 model has four blocks, width 128, four heads, a GELU feed-forward network, learned absolute positions, tied input/output embeddings, and 1,088,256 parameters. It already uses AdamW, weight decay, a 100-update warmup, cosine learning-rate decay, and gradient clipping; those are not new contributions of this work.

The final student has **six blocks, width 256, eight heads, SwiGLU hidden width 688, RoPE, and 5,290,048 trainable parameters**. Residual dropout is 0.3. It was trained from random initialization with seed 17 for **40,000 optimizer updates**, effective batch 32 and context 256, presenting **327,680,000 training targets** under sampling with replacement. The student uses FP32, AdamW beta values (0.9, 0.999), matrix/embedding weight decay 0.1 with norm/bias decay 0, and a complete cosine schedule with peak learning rate 0.0014 and a 0.1-peak floor. The only training-text-derived auxiliary asset is the bigram table described below.

We average the floating-point weights of **33 checkpoints at steps 24,000, 24,500, ..., 40,000**, all from this one run. There is no ensemble of 33 models at inference. We calibrate the averaged student's distribution by dividing its log probabilities by **T = 1.19** and renormalizing with softmax.

For the current observed token `u`, `C(u,v)` counts adjacent train tokens `u,v`, and `C(u)` is the sum of the row. The unigram prior is `p_uni(v) = (C(v)+1)/(N+2048)`, with `N` train tokens. With prespecified smoothing strength **beta = 32**, the auxiliary probability is `Q(v|u) = [C(u,v) + 32*p_uni(v)]/[C(u)+32]`. The final probability is **`0.98*P_T(v|prefix) + 0.02*Q(v|u)`**.

The table is a positive, row-normalized **2048 x 2048** FP32 buffer (**16 MiB**) embedded in the checkpoint. The train corpus has 3,613,343 encoded tokens and 298,024 observed adjacent pairs. Neither validation nor test targets enter the table. At scoring time the lookup uses only `u`, which is already observed before predicting the next token. The table is an extra inference asset, and its cost is included in the resource figures.

## 3. Controlled comparisons and ablations

### 3.1 Original baseline at equal processed targets

We first tested E0 against E3, a near-parameter-matched combination of SwiGLU and grouped weight decay (1,093,056 parameters, **0.44%** more than E0). At each budget, E0 and E3 were independently trained from random initialization with seeds 17, 29, and 43, matching sampled windows and the learning-rate schedule within each seed/budget pair. The table gives mean official CPU FP32 **validation** BPB at the fixed last update; the delta is the mean paired E3–E0 difference.

| Updates | Targets per run | E0 mean BPB | E3 mean BPB | Paired delta |
|---:|---:|---:|---:|---:|
| 1,200 | 9,830,400 | 2.074731 | 2.004174 | -0.070558 |
| 2,500 | 20,480,000 | 1.868128 | 1.813965 | -0.054163 |
| 4,000 | 32,768,000 | 1.774722 | 1.759765 | -0.014957 |

All nine within-budget paired differences favored E3. The advantage shrank as budgets grew; at 4,000 updates the mean paired difference was small relative to variation across only three seeds. These experiments establish a benefit of the **combination at matched training targets**, not a benefit at matched FLOPs or a claim that its 4,000-update advantage is statistically settled. The different update budgets are separate complete cosine schedules, so their cross-budget differences cannot be attributed solely to extra updates. In the 1,200-update two-by-two experiment, SwiGLU alone changed mean BPB from 2.074731 to 2.010064; grouped decay alone gave a smaller change from 2.074731 to 2.067937. This supports SwiGLU as the main early-budget contributor without isolating the gate from the rest of the feed-forward replacement. The source records are [Round 1](../evidence/round1-findings.md) and its [equal-budget extension](../evidence/round1-extension-findings.md).

### 3.2 Residual-dropout mechanism

For the later 5.29M-parameter CAP architecture, the cleanest regularization ablation fixed **seed 17, 12,000 updates, 98,304,000 targets per arm, the sampled-window stream, and the entire learning-rate stream**. Only residual dropout changed. Official validation BPB at the *same final update* was **2.020695** for dropout 0 and **1.541182** for dropout 0.2, a paired difference of **-0.479514 BPB**. The no-dropout run's own best checkpoint was much earlier (step 3,000, 1.671931 BPB), while the dropout run was still improving at step 12,000. The unequal-step best-checkpoint comparison is not used as the controlled effect. This ablation shows that residual dropout can prevent severe long-schedule overfitting in this architecture; it does **not** isolate the exact effect of the final dropout 0.3 setting or prove the result holds across seeds. A second seed was run for dropout 0.2 only. See the [paired ablation record](../evidence/dropout-findings.md).

### 3.3 Learning rate and the selected inference pipeline

At 40,000 updates and seed 17, the 0.0007, 0.0010, and 0.0014 peak-learning-rate arms used the same sampled windows and configuration apart from LR. On the raw final checkpoint their BPBs were **1.4997903, 1.5024546, and 1.5029799**: the lowest LR scored best. After a 17-checkpoint average and temperature calibration, the same order became **1.4707750, 1.4681605, and 1.4670057**: the highest LR scored best. Checkpoint averaging helped the high-LR trajectory more, reversing the ranking. This is an observed interaction on one seed, not evidence that increasing LR outside the tested range will continue to help.

The following sequential **validation** ablation uses the eventual 0.0014 trajectory. Probe values are shown to seven decimals; the final row was independently reproduced by the official frozen predictor. A probe shares the evaluator's split and FP32 prediction semantics, but its wall time is **not** the final scoring-time measure.

| Inference variant | Validation BPB | Change from preceding row |
|---|---:|---:|
| Final step 40,000, raw, no averaging | 1.5029799 | - |
| 33-checkpoint average, raw T=1, no bigram | 1.4922102 | -0.0107697 |
| Same average, T=1.19, no bigram | 1.4665408 | -0.0256694 |
| Same average and temperature, train-only bigram alpha=0.02 | **1.4647974** | **-0.0017434** |

The last component was tested on a fixed alpha grid of 0, 0.005, 0.01, 0.02, and 0.05. BPB was 1.4665408, 1.4654146, 1.4650006, **1.4647974**, and 1.4663321 respectively. The alpha-0 probe matched the temperature-only probe to about 0.0000001 BPB, and the frozen implementation matched the selected bigram probe **exactly** in reported BPB and NLL. Thus the measured increment is attributable to adding this fixed table in the implemented predictor, subject to validation selection bias. The smoothing constant beta and add-one prior were fixed before this sweep; they were not optimized on validation. A within-window unigram-copy probe also improved BPB by 0.0010452 but was not put into the deliverable because the bigram probe improved more under the same student distribution. Continuous cache and dynamic evaluation were not tested, so no conclusion is drawn about them. The [post-processing record](../evidence/posthoc-findings.md) and [bigram probe](../evidence/bigram-probe.json) give the underlying rows.

## 4. Frozen result and inference cost

The frozen checkpoint SHA-256 is **`2145fdb0acf4e07c678335fd6f8a88edb39ede72dbe5039c804a85a7e0ea4169`**. The student weights are bitwise identical, tensor for tensor, to the averaged base; freezing adds the declared temperature and mixture configuration and the train-derived bigram buffer. All 78 unit tests passed; six freeze/resource audit checks passed with an empty failure list. The supplied `evaluate.py`, data, tokenizer, and window logic were not modified. The [frozen validation score](../evidence/final-validation.json), [resource audit](../evidence/final-resource-check.json), and [test repetitions](../evidence/final-test-scoring.json) are included with the release.

| Split and measure | E0 anchor | Frozen predictor | Limit / interpretation |
|---|---:|---:|---|
| Validation BPB, official CPU FP32 | 1.7793456840 | **1.4647973939** | 376,599 targets; E0 anchor is a 4,000-update model, not an equal-target final-model control |
| Full-test BPB, official CPU FP32 | 1.8057584679 | **1.4850591242** | 428,405 targets; final candidate frozen before test |
| Paired validation scoring-time ratio, median of three | 1x | **3.8861x** | <=5x |
| Paired test scoring-time ratio, median of three | 1x | **3.6931x** | <=5x |
| Peak evaluation working set | - | **1.8706 GiB validation; 1.871 GiB test** | <=4 GiB |
| All uncompressed inference assets | - | **40,203,767 bytes = 38.337 MiB** | <=64 MiB |

The single checkpoint is 40,067,992 bytes (38.208 MiB), including the 16 MiB bigram table; the asset total additionally counts tokenizer and required model code. Validation timing used three interleaved E0-candidate repetitions, with per-pair ratios 3.8861, 3.8951, and 3.8801. Test timing likewise used three interleaved pairs; the candidate's BPB and E0's BPB were each identical across the three repetitions. The test score is **0.020262 BPB higher** than validation; these different splits must not be subtracted to estimate a method effect. The test anchor is a descriptive benchmark, not an equal-training-budget control.

## 5. Search cost and failed directions

Training targets count *presentations* under random window sampling with replacement, not unique tokens or epochs. The final student itself used 327,680,000 targets. Counting **every completed formal training run**, including the initial local E0 reproduction, the search consumed **4,084,531,200 target presentations**:

| Completed formal runs | Target presentations |
|---|---:|
| Initial E0 plus three-budget E0/E1/E2/E3 comparisons | 447,283,200 |
| P1/P2 architecture experiments and P3A combination | 294,912,000 |
| 27 September capacity/long-budget experiments | 589,824,000 |
| P4 dropout arms and seed-29 replication | 294,912,000 |
| P5 20,000-update student | 163,840,000 |
| H learning-rate arms | 327,680,000 |
| 28 September dropout/LR and two 30,000-update seeds | 819,200,000 |
| 20,000-update teacher and 40,000-update student | 491,520,000 |
| 40,000-update low/high-LR arms | 655,360,000 |
| Post-processing | **0 training targets** |

A separate **aborted** dropout-0.3 run logged through step 15,800 before termination: approximately **129,433,600 additional presented targets**. It was not resumed or used for model selection. Including it gives at least **4,213,964,800** observed search target presentations, before small development smoke tests and throughput probes. Those checks included a four-step random teacher and four-step CE/KD loading-and-resume checks; they were engineering tests, not quality comparisons. Reported wall time varies strongly with heat and machine load: the two 40,000-step LR arms alone took **17,552 seconds (4.876 h)** of training-process wall time, while the teacher/L40 sequence took about **3.44 h** including recorded scoring subprocesses. The post-processing round added about **207 s** of stage-A probes, **106 s** for table construction and stage-B probing, and **271 s** for formal validation freeze/scoring, without student training. These subprocess totals are not an end-to-end project wall-clock total. The [experiment ledger](../evidence/experiment-ledger.md) records the completed arms and the aborted run.

We also trained a 14.99M-parameter teacher for 20,000 updates (163,840,000 targets, already included above). Its prespecified calibrated validation BPB was **1.537464**, worse than the then-best student at **1.474099**. Under the preregistered quality gate, no full CE/KD student pair was launched; therefore **there is no measured distillation-versus-CE result**, and the experiment does not show that distillation in general fails. The teacher's raw diagnostic curve rose after its best early checkpoint, consistent with overfitting in this particular recipe but not a single-factor causal diagnosis.

## 6. Interpretation and limitations

The strongest causal comparisons here are the within-budget, matched-stream baseline pairs and the single-seed, same-target dropout ablation. The final **1.4647974 validation BPB** is a selected minimum after multiple architecture, schedule, checkpoint-window, temperature, and mixture probes. Its **0.0022082 BPB** advantage over the preceding frozen 17-checkpoint, T=1.2 candidate is measured on **one seed**. An observed cross-seed difference for a related recipe was about **0.0022 BPB**, similar to that advantage; it is not a confidence interval or an estimate of the selected model's generalization gap. The bigram-grid winner alpha=0.02 is only the best of five tested values. No second seed, unused validation holdout, or test-guided model search supports a broader improvement claim. The table's incremental benefit also does not identify whether it helps through local phrase statistics, frequent-token priors, or another mechanism; that requires a different ablation.

The last frozen predictor was selected entirely with validation. Full-test **1.4850591242** is reported as the score for this already selected checkpoint, not compared with other candidates to make a new choice. The resource margins are measured on this machine and the course's CPU FP32 evaluator; timing should be reproduced with the same paired protocol on another machine if challenged.

## 7. Reproduction and evidence

The release packages the supplied fixed scorer, tokenizer, and data together with the final model implementation. The matching checkpoint bundle is distributed separately and scores the model **without retraining**: once its `checkpoint.pt` is placed beside the release code's `evaluate.py`, the core command is:

```powershell
python evaluate.py --checkpoint checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The release README supplies the exact environment and bundle-location instructions. Check the downloaded checkpoint against the SHA-256 above before scoring. The published report tables are intentionally self-contained: the large local `runs/` history is not needed to execute the predictor. A selected evidence subset accompanies the release, while the complete original metrics, process logs, and source snapshots remain in the local experiment archive.

The release includes the selected [equal-target records](../evidence/round1-findings.md),
[dropout ablation](../evidence/dropout-findings.md), [post-processing record](../evidence/posthoc-findings.md),
[full experiment ledger](../evidence/experiment-ledger.md),
[frozen validation score](../evidence/final-validation.json),
[resource audit](../evidence/final-resource-check.json), and
[full-test timing repetitions](../evidence/final-test-scoring.json).
The [evidence index](../evidence/README.md) maps these selected files to the complete local research archive.

The student used **Codex and Claude Code** substantively for code development, experiment orchestration, auditing, and drafting. The student remains responsible for the experimental decisions, the correctness of the released bundle, and the claims in this report.
