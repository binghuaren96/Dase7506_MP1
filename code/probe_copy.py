r"""Validation-only causal unigram-cache probe for one frozen checkpoint.

For position t in each independent window, q_t(v) counts v only in observed
inputs x[0:t+1], divided by t+1. Probe (1-alpha)P_model + alpha*q for the fixed
alpha grid. Targets are used only after q is built, to calculate likelihoods.
This is a research probe, not a change to the course evaluator or submitted model.

``--temperature`` tempers the base model before the mixture, so the probe measures
the incremental value of the cache on top of an already temperature-calibrated
distribution. At the default ``1.0`` the tempering step is skipped entirely and
every row is bit-identical to the untempered probe.

PowerShell example from code/:
  .\.venv\Scripts\python.exe probe_copy.py --checkpoint runs\p4-cap-drop02-12k-20260927\CAP-p02-s17\checkpoint.pt --output runs\copy-probe\cap-p02-s17.json

The alpha=0 row follows evaluate.score's FP64 per-window reduction exactly and
is checked by a separate call to evaluate.score on the same validation tokens.
That check is only asserted at temperature 1.0, where the official scorer is the
right reference; at other temperatures alpha=0 is the calibrated score instead.
The probe grid time is not deployment inference time.
"""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from common import PROTOCOL, ROOT, setup, sha, windows
from evaluate import score as official_score
from probe_posthoc import checked_log_probs, load_checkpoint, validation_data


ALPHAS = (0.0, 0.02, 0.05, 0.10, 0.20)
VOCAB = 2048


def tempered_log_probs(logp, temperature):
    """Return log_softmax(logits/T) given log probabilities.

    softmax(logp/T) equals softmax(logits/T) because subtracting the log-normalizer
    before dividing by T is undone by the second normalization. At T=1 the transform
    is skipped rather than applied, so the untempered rows stay bit-identical to
    evaluate.score and to the earlier probe revisions.
    """
    if temperature == 1.0:
        return logp
    if not temperature > 0:
        raise ValueError('temperature must be positive.')
    return torch.log_softmax(logp / temperature, dim=-1)


def causal_unigram_cache(x, vocab_size=VOCAB):
    """Return normalized [batch,time,vocab] q using x only, resetting per row."""
    if x.ndim != 2 or x.dtype != torch.long:
        raise ValueError('x must be a two-dimensional long tensor.')
    if x.numel() == 0 or x.min().item() < 0 or x.max().item() >= vocab_size:
        raise ValueError('x contains an out-of-vocabulary token or is empty.')
    counts = torch.zeros((*x.shape, vocab_size), dtype=torch.float32, device=x.device)
    counts.scatter_(2, x.unsqueeze(-1), 1.0)
    counts = counts.cumsum(dim=1)
    return counts / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)


def mixture_target_losses(model_logp, cache, targets, alpha):
    """NLL of normalized probability mixture; alpha=0 is evaluator-exact."""
    if not 0 <= alpha < 1:
        raise ValueError('alpha must lie in [0, 1).')
    safe_targets = targets.clamp_min(0).unsqueeze(-1)
    log_p = model_logp.gather(-1, safe_targets).squeeze(-1)
    q = cache.gather(-1, safe_targets).squeeze(-1)
    if alpha == 0.0:
        losses = -log_p
    else:
        losses = -torch.logaddexp(log_p + math.log1p(-alpha),
                                  q.log() + math.log(alpha))
    losses.masked_fill_(targets == -100, 0)
    return losses


def append_window_sums(row, losses, present, absent):
    """Match evaluate.score's window order for total NLL and both slices."""
    for key, values in (
            ('nll_nats', losses),
            ('present_nll_nats', losses.masked_fill(~present, 0)),
            ('absent_nll_nats', losses.masked_fill(~absent, 0))):
        row[key] += sum(values.double().sum(-1).cpu().tolist())


@torch.no_grad()
def score_candidates(model, tokens, byte_count, alphas=ALPHAS, temperature=1.0):
    rows = {f'{alpha:g}': {'alpha': alpha, 'nll_nats': 0.0,
                          'present_nll_nats': 0.0, 'absent_nll_nats': 0.0}
            for alpha in alphas}
    counts = {'targets': 0, 'prefix_present': 0, 'prefix_absent': 0}
    derivative_nats = 0.0
    previous_mode = model.training
    model.eval()
    started = time.perf_counter()
    try:
        for x, y in windows(tokens, batch_size=32, context=256):
            # q is formed from x alone. The following gather only asks whether
            # the observed target was in the already-built causal distribution.
            logp = tempered_log_probs(checked_log_probs(model, x), temperature)
            cache = causal_unigram_cache(x)
            q_target = cache.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            valid = y != -100
            present = valid & (q_target > 0)
            absent = valid & ~present
            counts['targets'] += valid.sum().item()
            counts['prefix_present'] += present.sum().item()
            counts['prefix_absent'] += absent.sum().item()

            log_p_target = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            ratio = torch.zeros_like(q_target, dtype=torch.float64)
            ratio[present] = (q_target[present].double().log() -
                              log_p_target[present].double()).exp()
            if not torch.isfinite(ratio[present]).all():
                raise ValueError('Non-finite q(y)/p(y) in alpha=0 derivative.')
            derivative_nats += (valid.double() - ratio).sum().item()

            for alpha in alphas:
                losses = mixture_target_losses(logp, cache, y, alpha)
                append_window_sums(rows[f'{alpha:g}'], losses, present, absent)
    finally:
        model.train(previous_mode)
    elapsed = time.perf_counter() - started
    if counts['targets'] != counts['prefix_present'] + counts['prefix_absent']:
        raise AssertionError('Present and absent slices do not partition all targets.')

    baseline = rows['0']
    for row in rows.values():
        row['bpb'] = row['nll_nats'] / math.log(2) / byte_count
        row['delta_bpb_vs_alpha0'] = (row['nll_nats'] - baseline['nll_nats']) / math.log(2) / byte_count
        row['token_ppl'] = math.exp(row['nll_nats'] / counts['targets'])
        row['slice_deltas_vs_alpha0'] = {}
        for label, key in (('prefix_present', 'present_nll_nats'),
                           ('prefix_absent', 'absent_nll_nats')):
            delta = row[key] - baseline[key]
            row['slice_deltas_vs_alpha0'][label] = {
                'delta_nll_nats': delta,
                'delta_mean_token_nll_nats': delta / counts[label] if counts[label] else None,
                'delta_contribution_to_total_bpb': delta / math.log(2) / byte_count,
            }

    return {
        'rows': rows, 'counts': counts, 'utf8_bytes': byte_count,
        'temperature': temperature, 'alphas': list(alphas),
        'baseline_slices': {
            label: {'targets': counts[label], 'target_share': counts[label] / counts['targets'],
                    'nll_nats': baseline[key],
                    'mean_token_nll_nats':
                        baseline[key] / counts[label] if counts[label] else None}
            for label, key in (('prefix_present', 'present_nll_nats'),
                               ('prefix_absent', 'absent_nll_nats'))},
        'derivative_at_alpha0': {
            'mean_nll_nats_per_target': derivative_nats / counts['targets'],
            'total_nll_nats_per_unit_alpha': derivative_nats,
            'bpb_per_unit_alpha': derivative_nats / math.log(2) / byte_count,
            'formula': 'mean over valid targets of 1 - q_t(y_t)/P_model(y_t)',
        },
        'probe_grid_seconds': elapsed,
        'note': 'Grid timing includes every alpha candidate and cache construction; '
                'measure final inference separately.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--alpha', type=float, action='append',
                        help='Repeat for a custom alpha grid; 0 is always included.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temper the base model before mixing. 1.0 reproduces the '
                             'untempered probe exactly.')
    parser.add_argument('--temperature-probe', type=Path,
                        help='probe_posthoc.py output at the same temperature; when given, '
                             'the alpha=0 row is required to match its T row.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and positive.')
    alphas = sorted({float(value) for value in (args.alpha or ALPHAS)} | {0.0})
    if not all(math.isfinite(value) and 0.0 <= value < 1.0 for value in alphas):
        parser.error('--alpha must be finite and lie in [0, 1).')
    if args.output.exists():
        parser.error(f'Refusing to overwrite existing result: {args.output}')

    device, precision = setup('cpu', 'fp32', args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    tokens, byte_count = validation_data()
    result = score_candidates(model, tokens, byte_count, alphas, args.temperature)

    # Independent validation-only check against the fixed scorer. It receives
    # exactly the same tokens and never calls common.load_data or opens test.
    reference = official_score(model, tokens, byte_count, device, precision)
    reference.pop('window_nll_nats')
    alpha0 = result['rows']['0']
    counts_agree = (result['counts']['targets'] == reference['targets'] and
                    byte_count == reference['utf8_bytes'])
    # The fixed scorer is only the right reference for the untempered model.
    exact = counts_agree and (alpha0['nll_nats'] == reference['nll_nats'] and
                              alpha0['bpb'] == reference['bpb'])
    tempered_baseline = None
    if args.temperature != 1.0:
        if not counts_agree:
            raise ValueError('Target count disagrees with evaluate.score.')
        if args.temperature_probe is not None:
            probe = json.loads(args.temperature_probe.read_text(encoding='utf-8'))
            tempered_baseline = probe['rows'][f'model_1:T={args.temperature:g}']['bpb']
            # Same quantity as the probe's T row, reached through log_softmax instead of
            # logsumexp-minus-target. The two formulations agree mathematically but not
            # bitwise: measured on this model they differ by ~2.6e-8 BPB over 12 windows
            # (max 3.1e-6 nats per token), because PyTorch reduces the two kernels
            # differently in float32. 1e-6 is ~10x the full-validation gap and still
            # catches any real formula error.
            if abs(tempered_baseline - alpha0['bpb']) > 1e-6:
                raise ValueError('alpha=0 does not match the temperature probe at '
                                 f'T={args.temperature:g}.')
    result.update(
        protocol=PROTOCOL, split='validation', device='cpu', precision=precision,
        threads=args.threads, checkpoint=checkpoint, alphas=alphas,
        requested_temperature=args.temperature,
        best_alpha=min(alphas, key=lambda alpha: result['rows'][f'{alpha:g}']['bpb']),
        alpha0_exact_evaluate_parity=exact,
        untempered_evaluate_baseline_bpb=reference['bpb'],
        alpha0_matches_temperature_probe_bpb=tempered_baseline,
        evaluate_baseline={key: reference[key] for key in
                           ('bpb', 'nll_nats', 'targets', 'utf8_bytes', 'seconds')},
        probe_sha256=sha(Path(__file__)), evaluator_sha256=sha(ROOT / 'evaluate.py'),
        common_sha256=sha(ROOT / 'common.py'),
        tokenizer_sha256=sha(ROOT / 'data/tokenizer.json'),
        validation_sha256=sha(ROOT / 'data/wikitext_validation.txt'))
    if args.temperature == 1.0 and not exact:
        raise ValueError('alpha=0 did not exactly reproduce evaluate.score on validation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()),
                      'baseline_bpb': alpha0['bpb'],
                      'alpha0_exact_evaluate_parity': exact,
                      'best_alpha': result['best_alpha'],
                      'best_bpb': result['rows'][f'{result["best_alpha"]:g}']['bpb'],
                      'probe_grid_seconds': result['probe_grid_seconds']}, indent=2))


if __name__ == '__main__':
    main()
