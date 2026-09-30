r"""Validation-only train-built bigram interpolation probe for one frozen checkpoint.

The bigram table is counted from the course train split with the fixed tokenizer and
smoothed by the train unigram prior at the fixed strength beta=32, so every row is a
strictly positive normalized distribution over the 2048 output tokens. The prior itself
carries an add-one pseudocount because the fixed vocabulary contains 84 tokens the train
split never uses; without it those rows would keep a zero entry and the declared
"positive probability for all 2048 tokens" would not hold. For position t in each
independent window the probe mixes

    p = (1 - alpha) * P_student(. | x[0:t+1]) + alpha * p_bigram(. | x_t)

Both components condition on observed inputs only. Validation targets are read after
the mixture is formed and are used solely to compute likelihoods. The alpha grid is
fixed before the run; no table structure or smoothing strength is chosen on validation.
This is a research probe, not the course evaluator or a submitted model.

PowerShell example from code/:
  .\.venv\Scripts\python.exe probe_bigram.py --checkpoint runs\...\checkpoint.pt ^
      --temperature 1.19 --temperature-probe runs\...\L40hi-k33.json ^
      --output runs\posthoc-20260928\bigram\L40hi-k33.json
"""

import argparse
import json
import math
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from common import PROTOCOL, ROOT, setup, sha, windows
from evaluate import score as official_score
from probe_copy import mixture_target_losses, tempered_log_probs
from probe_posthoc import checked_log_probs, load_checkpoint, validation_data


VOCAB = 2048
BETA = 32.0
DEFAULT_ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.05)


def train_tokens():
    """Encode the train split with the fixed tokenizer without opening validation or test."""
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    tokenizer_path = ROOT / 'data/tokenizer.json'
    train_path = ROOT / 'data/wikitext_train.txt'
    for path in (tokenizer_path, train_path):
        if sha(path) != manifest['sha256'][path.name]:
            raise ValueError(f'Changed benchmark file: {path.name}')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw = train_path.read_bytes()
    ids = tokenizer.encode(raw.decode('utf-8')).ids
    return torch.tensor(ids, dtype=torch.long), len(raw)


def bigram_table(tokens, beta=BETA, vocab=VOCAB):
    """Return (table, metadata); row u is p(.|u) smoothed by the train unigram prior."""
    if tokens.numel() < 2:
        raise ValueError('The train split is too short to count bigrams.')
    if tokens.min().item() < 0 or tokens.max().item() >= vocab:
        raise ValueError('A train token id lies outside the fixed vocabulary.')
    if not beta > 0:
        raise ValueError('beta must be positive.')
    started = time.perf_counter()
    pairs = tokens[:-1] * vocab + tokens[1:]
    counts = torch.bincount(pairs, minlength=vocab * vocab).view(vocab, vocab).double()
    unigram = torch.bincount(tokens, minlength=vocab).double()
    unigram_prior = (unigram + 1.0) / (unigram.sum() + vocab)
    table = ((counts + beta * unigram_prior.unsqueeze(0)) /
             (counts.sum(dim=1, keepdim=True) + beta)).float()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(table).all() or not bool((table > 0).all()):
        raise ValueError('Every smoothed entry must be finite and strictly positive.')
    if not torch.allclose(table.sum(-1), torch.ones(vocab), atol=1e-5, rtol=0):
        raise ValueError('Smoothed bigram rows must be normalized.')
    metadata = {
        'beta': beta, 'vocab': vocab, 'unigram_pseudocount': 1.0,
        'train_tokens': tokens.numel(),
        'table_shape': [vocab, vocab], 'table_bytes': table.numel() * table.element_size(),
        'build_seconds': elapsed, 'observed_pairs': int((counts > 0).sum().item()),
        'min_probability': table.min().item(), 'max_probability': table.max().item(),
    }
    return table, metadata


@torch.no_grad()
def score_candidates(model, table, tokens, byte_count, alphas, temperature=1.0):
    """Mixture NLL per alpha; alpha=0 follows evaluate.score's window reduction exactly."""
    rows = {f'{alpha:g}': {'alpha': alpha, 'nll_nats': 0., 'window_count': 0}
            for alpha in alphas}
    previous_mode = model.training
    model.eval()
    started = time.perf_counter()
    target_count = 0
    try:
        for x, y in windows(tokens, batch_size=32, context=256):
            logp = tempered_log_probs(checked_log_probs(model, x), temperature)
            # Row x_t holds p_bigram(. | x_t): every position is conditioned on its own
            # observed input token, so no window or row carries future information.
            bigram = table[x]
            target_count += (y != -100).sum().item()
            for alpha in alphas:
                losses = mixture_target_losses(logp, bigram, y, alpha)
                row = rows[f'{alpha:g}']
                per_window = losses.double().sum(-1).cpu().tolist()
                row['nll_nats'] += sum(per_window)
                row['window_count'] += len(per_window)
    finally:
        model.train(previous_mode)
    elapsed = time.perf_counter() - started
    baseline = rows['0']['nll_nats']
    for row in rows.values():
        row['bpb'] = row['nll_nats'] / math.log(2) / byte_count
        row['delta_bpb_vs_alpha0'] = (row['nll_nats'] - baseline) / math.log(2) / byte_count
        row['token_ppl'] = math.exp(row['nll_nats'] / target_count)
        row['targets'] = target_count
        row['utf8_bytes'] = byte_count
    return {'rows': rows, 'probe_seconds': elapsed,
            'note': 'probe_seconds includes every alpha candidate; measure final '
                    'inference separately. Building the table costs train tokens, '
                    'so this is not a zero-cost weight-only postprocessing step.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--alpha', type=float, action='append',
                        help='Repeat for a custom alpha grid; 0 is always included.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temper the student before mixing. 1.0 reproduces the '
                             'untempered baseline exactly.')
    parser.add_argument('--temperature-probe', type=Path,
                        help='probe_posthoc.py output at the same temperature; required '
                             'when --temperature is not 1.0 so that alpha=0 is checked.')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be finite and positive.')
    alphas = sorted({float(value) for value in (args.alpha or DEFAULT_ALPHAS)} | {0.0})
    if not all(math.isfinite(value) and 0.0 <= value < 1.0 for value in alphas):
        parser.error('--alpha must be finite and lie in [0, 1).')
    if args.temperature != 1.0 and args.temperature_probe is None:
        parser.error('--temperature-probe is required when --temperature is not 1.0.')
    if args.output.exists():
        parser.error(f'Refusing to overwrite existing result: {args.output}')

    device, precision = setup('cpu', 'fp32', args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    tokens, byte_count = validation_data()
    encode_started = time.perf_counter()
    train_ids, train_byte_count = train_tokens()
    table, metadata = bigram_table(train_ids)
    metadata.update(tokenize_seconds=time.perf_counter() - encode_started,
                    train_utf8_bytes=train_byte_count,
                    train_sha256=sha(ROOT / 'data/wikitext_train.txt'))
    result = score_candidates(model, table, tokens, byte_count, alphas, args.temperature)

    # Independent validation-only reference. It receives the same validation tokens and
    # never calls common.load_data, so the test split stays closed.
    reference = official_score(model, tokens, byte_count, device, precision)
    reference.pop('window_nll_nats')
    alpha0 = result['rows']['0']
    counts_agree = (alpha0['targets'] == reference['targets'] and
                    byte_count == reference['utf8_bytes'])
    if not counts_agree:
        raise ValueError('Target count disagrees with evaluate.score.')
    tempered_baseline = None
    if args.temperature == 1.0:
        if alpha0['nll_nats'] != reference['nll_nats'] or alpha0['bpb'] != reference['bpb']:
            raise ValueError('alpha=0 did not exactly reproduce evaluate.score on validation.')
    else:
        probe = json.loads(args.temperature_probe.read_text(encoding='utf-8'))
        tempered_baseline = probe['rows'][f'model_1:T={args.temperature:g}']['bpb']
        # Same quantity reached through log_softmax rather than logsumexp-minus-target;
        # the float32 kernels differ by ~1e-7 BPB, so 1e-6 still catches a formula error.
        if abs(tempered_baseline - alpha0['bpb']) > 1e-6:
            raise ValueError('alpha=0 does not match the temperature probe at '
                             f'T={args.temperature:g}.')
    result.update(
        protocol=PROTOCOL, split='validation', device='cpu', precision=precision,
        threads=args.threads, checkpoint=checkpoint, alphas=alphas, bigram=metadata,
        requested_temperature=args.temperature,
        best_alpha=min(alphas, key=lambda alpha: result['rows'][f'{alpha:g}']['bpb']),
        alpha0_matches_temperature_probe_bpb=tempered_baseline,
        untempered_evaluate_baseline_bpb=reference['bpb'],
        evaluate_baseline={key: reference[key] for key in
                           ('bpb', 'nll_nats', 'targets', 'utf8_bytes', 'seconds')},
        probe_sha256=sha(Path(__file__)), evaluator_sha256=sha(ROOT / 'evaluate.py'),
        common_sha256=sha(ROOT / 'common.py'),
        tokenizer_sha256=sha(ROOT / 'data/tokenizer.json'),
        validation_sha256=sha(ROOT / 'data/wikitext_validation.txt'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                           encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()),
                      'baseline_bpb': alpha0['bpb'],
                      'requested_temperature': args.temperature,
                      'best_alpha': result['best_alpha'],
                      'best_bpb': result['rows'][f'{result["best_alpha"]:g}']['bpb'],
                      'table_bytes': metadata['table_bytes'],
                      'build_seconds': metadata['build_seconds'],
                      'probe_seconds': result['probe_seconds']}, indent=2))


if __name__ == '__main__':
    main()
