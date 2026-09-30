r"""Freeze the final predictor and verify it with the original course scorer.

Step 1 builds the delivery checkpoint by copying a trained checkpoint's weights and
adding ``temperature`` to its config. That transform is config-only: no parameter is
recomputed, nothing is retrained, and the source weights are copied unchanged, so the
frozen file differs from its base only by the recorded config key and the provenance
block. ``research_models.RoPEGPT`` skips the transform entirely at T=1, which is why
every earlier checkpoint keeps its exact recorded score.

Step 2 interleaves the original course baseline E0 with the frozen predictor over
repeated CPU FP32 validation passes under the same background load, so the 5x time
limit and the 4 GiB memory limit are judged from a paired measurement rather than
from two separately timed runs.

Validation only. ``evaluate.py`` opens every split through ``common.load_data``, but
``--split validation`` is the only split scored here.

PowerShell example (from code/):
  .\.venv\Scripts\python.exe freeze_final.py --base runs\p4b-posthoc-20260927\CAP-p02-s17-20k-avg19k-20k\checkpoint.pt --output runs\p4b-posthoc-20260927\final --temperature 1.2 --repeats 3
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import torch

from common import PROTOCOL, ROOT, sha
from probe_bigram import bigram_table, train_tokens
from run_round1 import measured_run


E0_ANCHOR = ROOT / 'runs/round1-extension-4000-20260926/E0-s17/checkpoint.pt'
RAM_LIMIT_BYTES = 4 * 1024 ** 3
ASSET_LIMIT_BYTES = 64 * 1024 ** 2
CPU_TIME_LIMIT = 5.
BIGRAM_IMPLEMENTATION = 'bigram_models'
STUDENT_SOURCES = ('research_models.py', 'student.py', 'model.py')
BIGRAM_SOURCES = ('bigram_models.py',)


def delivery_bigram_table(alpha):
    """Rebuild the train-only table the probes measured, for embedding in a checkpoint."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0. <= alpha < 1.:
        raise ValueError('bigram_alpha must lie in [0, 1).')
    tokens, train_bytes = train_tokens()
    table, record = bigram_table(tokens)
    record.update(alpha=alpha, train_sha256=sha(ROOT / 'data/wikitext_train.txt'),
                  train_utf8_bytes=train_bytes,
                  tokenizer_sha256=sha(ROOT / 'data/tokenizer.json'))
    return table, record


def inference_asset_bytes(checkpoint_bytes, implementation):
    """Uncompressed inference assets: the checkpoint, the tokenizer and the model code."""
    sources = STUDENT_SOURCES + (BIGRAM_SOURCES if implementation == BIGRAM_IMPLEMENTATION else ())
    return (checkpoint_bytes + (ROOT / 'data/tokenizer.json').stat().st_size +
            sum((ROOT / name).stat().st_size for name in sources)), sorted(sources)


def freeze(base_path, output_dir, temperature, bigram_alpha=None):
    """Copy ``base_path``'s weights into a delivery checkpoint carrying ``temperature``.

    With ``bigram_alpha`` the payload additionally embeds a freshly rebuilt train-only
    bigram table, so the student weights are still copied unchanged but the delivery
    predictor is no longer purely config-only: it carries an extra inference asset.
    """
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if not temperature > 0:
        raise ValueError('temperature must be positive.')
    base = torch.load(base_path, map_location='cpu', weights_only=True)
    for field in ('protocol', 'implementation', 'config', 'model', 'seed'):
        if field not in base:
            raise ValueError(f'Base checkpoint lacks {field}: {base_path}')
    if base['protocol'] != PROTOCOL:
        raise ValueError(f'Base checkpoint has the wrong protocol: {base_path}')
    if 'temperature' in base['config']:
        raise ValueError('Base checkpoint already carries a temperature; freeze that one instead.')
    destination = output_dir / 'checkpoint.pt'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    payload = dict(base)
    config = {**base['config'], 'temperature': temperature}
    record = {
        'method': 'config_only_temperature', 'weights_recomputed': False,
        'source': str(base_path.resolve()), 'source_sha256': sha(base_path),
        'temperature': temperature, 'protocol': PROTOCOL}
    if bigram_alpha is not None:
        table, table_record = delivery_bigram_table(bigram_alpha)
        config['bigram_alpha'] = bigram_alpha
        payload['implementation'] = BIGRAM_IMPLEMENTATION
        payload['model'] = {**base['model'], 'bigram_table': table}
        record = {
            'method': 'student_weights_plus_train_only_bigram',
            'weights_recomputed': False, 'student_weights_recomputed': False,
            'source': str(base_path.resolve()), 'source_sha256': sha(base_path),
            'temperature': temperature, 'bigram_alpha': bigram_alpha,
            'protocol': PROTOCOL, 'bigram': table_record}
    payload['config'] = config
    payload['freeze'] = record
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix='.checkpoint.', suffix='.tmp',
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination, payload


def scoring_plan(targets, output_dir, repeats):
    """One official output per target, plus every repeat for the paired timing ratios."""
    plan = []
    for rep in range(1, repeats + 1):
        for name, checkpoint in targets.items():
            official = rep == 1
            plan.append({
                'target': name, 'rep': rep, 'checkpoint': checkpoint, 'official': official,
                'output': (output_dir / name / 'validation_cpu_fp32.json' if official
                           else output_dir / 'cpu-scoring' / f'{name}-rep{rep}.json'),
                'log': (output_dir / f'{name}.cpu.log' if official
                        else output_dir / 'cpu-scoring' / f'{name}-rep{rep}.log')})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh directory for the frozen checkpoint and its evidence.')
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--bigram-alpha', type=float,
                        help='Embed a freshly rebuilt train-only bigram table and mix it '
                             'at this alpha; omit for the config-only temperature freeze.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', type=Path, default=E0_ANCHOR)
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory; do not overwrite earlier evidence.')
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if not args.skip_tests:
        measured_run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                     args.output / 'tests.log')
    frozen, payload = freeze(args.base, args.output, args.temperature, args.bigram_alpha)

    targets = {'E0-anchor': args.baseline,
               f'frozen-T{args.temperature:g}': frozen}
    (args.output / 'cpu-scoring').mkdir(parents=True, exist_ok=True)
    scoring = []
    for entry in scoring_plan(targets, args.output, args.repeats):
        measured = measured_run([sys.executable, 'evaluate.py', '--checkpoint', str(entry['checkpoint']),
                                 '--device', 'cpu', '--precision', 'fp32', '--threads', '4',
                                 '--split', 'validation', '--output', str(entry['output'])],
                                entry['log'])
        result = json.loads(entry['output'].read_text(encoding='utf-8'))
        scoring.append({'target': entry['target'], 'rep': entry['rep'], 'official': entry['official'],
                        'bpb': result['bpb'], 'seconds': result['seconds'],
                        'targets': result['targets'],
                        'cpu_command_seconds': measured['wall_seconds'],
                        'cpu_peak_working_set_bytes':
                            measured['peak_process_tree_working_set_upper_bound_bytes'],
                        'output': str(entry['output'])})
        (args.output / 'cpu-scoring.json').write_text(json.dumps(scoring, indent=2), encoding='utf-8')
        print('SCORE ' + json.dumps({'target': entry['target'], 'rep': entry['rep'],
                                     'bpb': result['bpb'], 'seconds': result['seconds']}), flush=True)

    baseline = [row for row in scoring if row['target'] == 'E0-anchor']
    frozen_rows = [row for row in scoring if row['target'] != 'E0-anchor']
    frozen_name = next(name for name in targets if name != 'E0-anchor')
    by_rep = {row['rep']: row for row in baseline}
    paired = [row['seconds'] / by_rep[row['rep']]['seconds'] for row in frozen_rows
              if row['rep'] in by_rep]
    official = next(row for row in frozen_rows if row['official'])
    official_bpb = json.loads(Path(official['output']).read_text(encoding='utf-8'))
    base_bytes = args.base.stat().st_size
    frozen_bytes = frozen.stat().st_size

    checks = []
    def check(name, ok, detail=None):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})

    written = torch.load(frozen, map_location='cpu', weights_only=True)
    original = torch.load(args.base, map_location='cpu', weights_only=True)
    extra_keys = set(written['model']) - set(original['model'])
    covered = set(original['model']) <= set(written['model'])
    weights_identical = covered and all(
        torch.equal(written['model'][name], original['model'][name]) for name in original['model'])
    check('frozen_student_weights_identical_to_base',
          weights_identical and extra_keys <= {'bigram_table'},
          {'entries': len(original['model']), 'extra_keys': sorted(extra_keys)})
    added = {'temperature'} | ({'bigram_alpha'} if args.bigram_alpha is not None else set())
    check('frozen_config_adds_only_declared_keys',
          {k: v for k, v in written['config'].items() if k not in added} == original['config'],
          {'added': sorted(added)})
    check('frozen_bpb_deterministic_across_reps',
          max(row['bpb'] for row in frozen_rows) - min(row['bpb'] for row in frozen_rows) < 1e-9)
    check('cpu_time_within_5x_by_paired_median',
          bool(paired) and statistics.median(paired) <= CPU_TIME_LIMIT,
          {'paired_by_rep': paired, 'paired_median': statistics.median(paired) if paired else None})
    check('ram_within_4gib',
          all((row['cpu_peak_working_set_bytes'] or 0) <= RAM_LIMIT_BYTES for row in scoring),
          max((row['cpu_peak_working_set_bytes'] or 0) for row in scoring))
    asset_bytes, asset_sources = inference_asset_bytes(frozen_bytes, payload['implementation'])
    check('asset_within_64mib', asset_bytes <= ASSET_LIMIT_BYTES,
          {'inference_asset_bytes': asset_bytes, 'checkpoint_bytes': frozen_bytes,
           'inference_sources': asset_sources})

    result = {
        'purpose': (f'Frozen final predictor: temperature {args.temperature:g}'
                    + (f' plus train-only bigram mix at alpha {args.bigram_alpha:g}'
                       if args.bigram_alpha is not None else ' (config-only)')
                    + f' on {args.base.name} in {args.base.parent.name}.'),
        'split': 'validation',
        'base_checkpoint': str(args.base.resolve()), 'base_sha256': sha(args.base),
        'base_bytes': base_bytes,
        'frozen_checkpoint': str(frozen.resolve()), 'frozen_sha256': sha(frozen),
        'frozen_bytes': frozen_bytes, 'inference_asset_bytes': asset_bytes,
        'temperature': args.temperature,
        'config': payload['config'], 'freeze': payload['freeze'],
        'e0_baseline_checkpoint': str(args.baseline.resolve()),
        'official_bpb': official_bpb['bpb'], 'official_targets': official_bpb['targets'],
        'cpu_scoring': scoring,
        'paired_seconds_ratio': {'by_rep': paired,
                                 'median': statistics.median(paired) if paired else None,
                                 'min_seconds_ratio':
                                     min(row['seconds'] for row in frozen_rows) /
                                     min(row['seconds'] for row in baseline)},
        'peak_working_set_bytes': max((row['cpu_peak_working_set_bytes'] or 0) for row in scoring),
        'round_seconds': time.perf_counter() - started,
        'audit': {'checks': checks, 'failed': [row for row in checks if not row['ok']]},
    }
    (args.output / 'resource-check.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'frozen': str(frozen), 'official_bpb': result['official_bpb'],
                      'paired_median_ratio': result['paired_seconds_ratio']['median'],
                      'peak_working_set_bytes': result['peak_working_set_bytes'],
                      'frozen_bytes': frozen_bytes,
                      'audit_failed': [row['name'] for row in result['audit']['failed']]},
                     indent=2, ensure_ascii=False))
    return 1 if result['audit']['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
