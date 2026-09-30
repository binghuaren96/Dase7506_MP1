"""Run the predeclared 2x2 matrix sequentially, with isolated logs and CPU measurements."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent


def measured_run(command, log_path):
    peak = 0
    started = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        observer = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                total = 0
                for child in [observer, *observer.children(recursive=True)]:
                    try:
                        memory = child.memory_info()
                        total += max(memory.rss, getattr(memory, 'peak_wset', 0))
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, total)
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
    row = {'command': command, 'returncode': proc.returncode,
           'wall_seconds': time.perf_counter()-started,
           'peak_process_tree_working_set_upper_bound_bytes': peak}
    log_path.with_suffix('.process.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'Command failed; see {log_path}')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[17])
    parser.add_argument('--experiments', nargs='+', choices=['E0', 'E1', 'E2', 'E3'], default=['E0', 'E1', 'E2', 'E3'])
    parser.add_argument('--steps', type=int, default=1200,
                        help='Training steps per run. This is also the LR-schedule horizon, so a '
                             'different value is a different recipe, not a longer run of the same one.')
    parser.add_argument('--eval-every', type=int, default=300,
                        help='Validation-diagnostic interval, applied identically to every run in the round.')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if min(args.steps, args.eval_every) < 1:
        parser.error('steps and eval-every must be positive')
    output = args.run_dir or ROOT/'runs'/('round1-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    for name in ['model.py', 'student.py', 'train.py', 'train_round1.py', 'run_round1.py',
                 'evaluate.py', 'common.py', 'requirements.txt', 'IMPROVEMENT_PLAN.md',
                 'summarize_round1.py', 'summarize_extension.py']:
        if (ROOT/name).exists():
            shutil.copy2(ROOT/name, snapshot/name)
    shutil.copytree(ROOT/'tests', snapshot/'tests', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(ROOT/'configs', snapshot/'configs')
    print(f'ROUND_DIR={output}', flush=True)
    measured_run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], output/'tests.log')
    rows = []
    for seed in args.seeds:
        for experiment in args.experiments:
            label = f'{experiment}-s{seed}'
            run_dir = output/label
            print(f'START {label}', flush=True)
            training = measured_run([sys.executable, 'train_round1.py', '--experiment', experiment,
                                     '--seed', str(seed), '--steps', str(args.steps),
                                     '--eval-every', str(args.eval_every),
                                     '--run-dir', str(run_dir)], output/f'{label}.train.log')
            evaluation = measured_run([sys.executable, 'evaluate.py', '--checkpoint', str(run_dir/'checkpoint.pt'),
                                       '--device', 'cpu', '--precision', 'fp32', '--threads', '4',
                                       '--split', 'validation'], output/f'{label}.cpu.log')
            metrics = json.loads((run_dir/'metrics.json').read_text(encoding='utf-8'))
            cpu = json.loads((run_dir/'validation_cpu_fp32.json').read_text(encoding='utf-8'))
            row = {'experiment': experiment, 'seed': seed, 'parameters': metrics['parameters'],
                   'steps': metrics['steps'], 'eval_every': args.eval_every,
                   'train_tokens': metrics['train_tokens'], 'train_seconds': metrics['train_seconds'],
                   'training_command_seconds': training['wall_seconds'],
                   'validation_bpb': cpu['bpb'], 'cpu_score_seconds': cpu['seconds'],
                   'cpu_command_seconds': evaluation['wall_seconds'],
                   'cpu_peak_working_set_bytes': evaluation['peak_process_tree_working_set_upper_bound_bytes'],
                   'checkpoint_bytes': (run_dir/'checkpoint.pt').stat().st_size,
                   'peak_gpu_allocated_gb': metrics['peak_allocated_gb'],
                   'training_window_sha256': metrics['training_window_sha256'],
                   'learning_rate_sha256': metrics['learning_rate_sha256'],
                   'diagnostic_window_sha256': metrics['diagnostic_window_sha256'],
                   'run_dir': str(run_dir)}
            rows.append(row)
            (output/'summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print('RESULT '+json.dumps(row), flush=True)
    for seed in args.seeds:
        paired = [row for row in rows if row['seed'] == seed]
        for key in ['training_window_sha256', 'learning_rate_sha256', 'diagnostic_window_sha256', 'train_tokens']:
            if len({row[key] for row in paired}) != 1:
                raise RuntimeError(f'Control mismatch: seed {seed}, {key}')
    print('COMPLETE: fixed-target controls verified; validation only.', flush=True)


if __name__ == '__main__':
    main()
