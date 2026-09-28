r"""Validation-only temperature and probability-mixture probe for frozen checkpoints.

This is a research probe, not the course evaluator or an inference submission.
It uses common.windows and the fixed tokenizer, but deliberately does not call
common.load_data because that helper also opens the test split. Each model sees
only the current causal input window. No validation targets enter either model.

Examples (PowerShell, from code/):
  .\.venv\Scripts\python.exe probe_posthoc.py --checkpoint runs\...\checkpoint.pt --output runs\posthoc\single.json
  .\.venv\Scripts\python.exe probe_posthoc.py --checkpoint runs\...\seed17.pt --checkpoint runs\...\seed29.pt --output runs\posthoc\pair.json

The T=1 and mixture endpoint rows use the original log probabilities and the
same per-window FP64 summation order as evaluate.score. A mixture weight w means
(1-w)*P(checkpoint 1) + w*P(checkpoint 2), never a weight average. Grid search
cost is reported separately; it is not a measurement of deployment inference.
"""

import argparse
import json
import math
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from common import PROTOCOL, ROOT, make_model, setup, sha, windows


DEFAULT_TEMPERATURES = (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)
DEFAULT_MIX_WEIGHTS = (0.00, 0.25, 0.50, 0.75, 1.00)


def validation_data():
    """Reproduce common.load_data()['validation'] without opening train or test."""
    manifest_path = ROOT / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    tokenizer_path = ROOT / 'data/tokenizer.json'
    validation_path = ROOT / 'data/wikitext_validation.txt'
    for path in (tokenizer_path, validation_path):
        if sha(path) != manifest['sha256'][path.name]:
            raise ValueError(f'Changed benchmark file: {path.name}')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    raw = validation_path.read_bytes()
    ids = tokenizer.encode(raw.decode('utf-8')).ids
    return torch.tensor(ids, dtype=torch.long), len(raw)


def checked_grid(values, *, name, low, high, endpoints):
    grid = sorted(set(float(value) for value in (*values, *endpoints)))
    if not all(math.isfinite(value) and low <= value <= high for value in grid):
        raise ValueError(f'{name} must be finite and lie in [{low}, {high}].')
    return grid


def checked_log_probs(model, x):
    logp = model.predict_log_probs(x).float()
    if logp.shape != (*x.shape, 2048) or not torch.isfinite(logp).all():
        raise ValueError('Return finite log probabilities of shape [batch, time, 2048].')
    if torch.logsumexp(logp, dim=-1).abs().max().item() > 1e-3:
        raise ValueError('The output is not a normalized probability distribution.')
    return logp


def temperature_target_losses(logp, y, temperature):
    """Token NLL after temperature scaling; T=1 exactly follows evaluate.score."""
    target = logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    if temperature == 1.0:
        losses = -target
    else:
        losses = torch.logsumexp(logp / temperature, dim=-1) - target / temperature
    losses.masked_fill_(y == -100, 0)
    return losses


def mixture_target_losses(first_logp, second_logp, y, second_weight):
    """Exact NLL for a normalized probability mixture at observed targets."""
    first_target = first_logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    second_target = second_logp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    if second_weight == 0.0:
        losses = -first_target
    elif second_weight == 1.0:
        losses = -second_target
    else:
        losses = -torch.logaddexp(first_target + math.log1p(-second_weight),
                                   second_target + math.log(second_weight))
    losses.masked_fill_(y == -100, 0)
    return losses


def append_window_sums(row, losses):
    # Keep evaluate.score's reduction order, including its short last window.
    per_window = losses.double().sum(-1).cpu().tolist()
    row['nll_nats'] += sum(per_window)
    row['window_count'] += len(per_window)


@torch.no_grad()
def score_candidates(models, tokens, byte_count, temperatures, mix_weights):
    if not 1 <= len(models) <= 2:
        raise ValueError('Supply one or two frozen models.')
    rows = {}
    for model_index in range(len(models)):
        for temperature in temperatures:
            key = f'model_{model_index + 1}:T={temperature:g}'
            rows[key] = {'kind': 'temperature', 'model': model_index + 1,
                         'temperature': temperature, 'nll_nats': 0., 'window_count': 0}
    if len(models) == 2:
        for weight in mix_weights:
            key = f'mixture:w2={weight:g}'
            rows[key] = {'kind': 'probability_mixture', 'second_weight': weight,
                         'nll_nats': 0., 'window_count': 0}

    previous_modes = [model.training for model in models]
    for model in models:
        model.eval()
    started = time.perf_counter()
    target_count = 0
    try:
        for x, y in windows(tokens, batch_size=32, context=256):
            logps = [checked_log_probs(model, x) for model in models]
            target_count += (y != -100).sum().item()
            for index, logp in enumerate(logps):
                for temperature in temperatures:
                    key = f'model_{index + 1}:T={temperature:g}'
                    append_window_sums(rows[key], temperature_target_losses(logp, y, temperature))
            if len(logps) == 2:
                for weight in mix_weights:
                    key = f'mixture:w2={weight:g}'
                    append_window_sums(rows[key], mixture_target_losses(*logps, y, weight))
    finally:
        for model, previous_mode in zip(models, previous_modes):
            model.train(previous_mode)
    elapsed = time.perf_counter() - started
    for row in rows.values():
        row['bpb'] = row['nll_nats'] / math.log(2) / byte_count
        row['token_ppl'] = math.exp(row['nll_nats'] / target_count)
        row['targets'] = target_count
        row['utf8_bytes'] = byte_count
    return {'rows': rows, 'probe_seconds': elapsed,
            'note': 'probe_seconds includes all grid candidates; measure final inference separately'}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError(f'Checkpoint belongs to a different course protocol: {path}')
    model, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model, {'path': str(path.resolve()), 'sha256': sha(path),
                   'implementation': checkpoint['implementation'],
                   'implementation_sha256': implementation_sha,
                   'config': checkpoint['config'], 'train_tokens': checkpoint.get('train_tokens')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True,
                        help='Repeat exactly twice to probe a probability mixture.')
    parser.add_argument('--temperature', type=float, action='append',
                        help='Repeat for a custom grid; T=1 is always included.')
    parser.add_argument('--mix-weight', type=float, action='append',
                        help='Weight on checkpoint 2; 0 and 1 are always included.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.checkpoint) <= 2:
        parser.error('Provide exactly one or two --checkpoint arguments.')
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if args.output.exists():
        parser.error(f'Refusing to overwrite existing result: {args.output}')
    temperatures = checked_grid(args.temperature or DEFAULT_TEMPERATURES,
                                name='temperature', low=0.01, high=100., endpoints=(1.0,))
    mix_weights = checked_grid(args.mix_weight or DEFAULT_MIX_WEIGHTS,
                               name='mix weight', low=0., high=1., endpoints=(0., 1.))
    device, precision = setup('cpu', 'fp32', args.threads)
    models_and_metadata = [load_checkpoint(path, device) for path in args.checkpoint]
    models = [item[0] for item in models_and_metadata]
    tokens, byte_count = validation_data()
    result = score_candidates(models, tokens, byte_count, temperatures, mix_weights)
    result.update(protocol=PROTOCOL, split='validation', device='cpu', precision=precision,
                  threads=args.threads, checkpoints=[item[1] for item in models_and_metadata],
                  temperatures=temperatures, mix_weights=mix_weights if len(models) == 2 else [],
                  probe_sha256=sha(Path(__file__)), evaluator_sha256=sha(ROOT/'evaluate.py'),
                  common_sha256=sha(ROOT/'common.py'),
                  tokenizer_sha256=sha(ROOT/'data/tokenizer.json'),
                  validation_sha256=sha(ROOT/'data/wikitext_validation.txt'))
    result['best_temperature_per_model'] = {
        f'model_{index + 1}': min((row for row in result['rows'].values()
                                  if row['kind'] == 'temperature' and row['model'] == index + 1),
                                 key=lambda row: row['bpb'])['temperature']
        for index in range(len(models))}
    if len(models) == 2:
        result['best_mixture_second_weight'] = min(
            (row for row in result['rows'].values() if row['kind'] == 'probability_mixture'),
            key=lambda row: row['bpb'])['second_weight']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()),
                      'best_temperature_per_model': result['best_temperature_per_model'],
                      'best_mixture_second_weight': result.get('best_mixture_second_weight'),
                      'baseline_bpb': [result['rows'][f'model_{index + 1}:T=1']['bpb']
                                       for index in range(len(models))],
                      'probe_seconds': result['probe_seconds']}, indent=2))


if __name__ == '__main__':
    main()
