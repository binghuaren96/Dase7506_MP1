"""Controlled MP1 experiments. Original train.py and fixed evaluator stay intact."""
import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import PROTOCOL, ROOT, device_metrics, load_data, make_model, setup, sha
from evaluate import score

EXPERIMENTS = {
    'E0': ('model', False, 'GELU / uniform decay'),
    'E1': ('student', False, 'SwiGLU / uniform decay'),
    'E2': ('model', True, 'GELU / grouped decay'),
    'E3': ('student', True, 'SwiGLU / grouped decay'),
}


def optimizer_groups(model, grouped):
    """Decay embeddings/matrices; optionally exclude LayerNorm and biases."""
    named = list(model.named_parameters())  # deduplicates the tied embedding
    no_decay_ids = set()
    if grouped:
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_ids.update(id(p) for p in module.parameters(recurse=False))
            bias = getattr(module, 'bias', None)
            if isinstance(bias, nn.Parameter):
                no_decay_ids.add(id(bias))
    regular = [(name, p) for name, p in named if id(p) not in no_decay_ids]
    excluded = [(name, p) for name, p in named if id(p) in no_decay_ids]
    groups, description = [], []
    for entries, decay in ((regular, .1), (excluded, 0.)):
        if entries:
            groups.append({'params': [p for _, p in entries], 'weight_decay': decay})
            description.append({'weight_decay': decay, 'names': [n for n, _ in entries],
                                'parameters': sum(p.numel() for _, p in entries)})
    return groups, description


@torch.no_grad()
def training_diagnostic(model, batches):
    previous = model.training
    model.eval()
    nll, targets = 0., 0
    for batch in batches:
        logp = model.predict_log_probs(batch[:, :-1]).float()
        nll += (-logp.gather(-1, batch[:, 1:].unsqueeze(-1)).squeeze(-1)).double().sum().item()
        targets += batch[:, 1:].numel()
    model.train(previous)
    return {'nll_nats': nll, 'targets': targets, 'mean_token_nll': nll / targets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=EXPERIMENTS, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=300)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1:
        parser.error('steps, batch-size and eval-every must be positive')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Use a fresh output directory')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    process_start = time.perf_counter()
    device, precision = setup(args.device, 'fp32', args.threads)
    torch.manual_seed(args.seed)
    data = load_data()
    config = json.loads((ROOT / 'configs/baseline.json').read_text())
    implementation, grouped, label = EXPERIMENTS[args.experiment]
    if implementation == 'student':
        config.update(ffn='swiglu', ffn_hidden=344)
    model, implementation_sha = make_model(implementation, config, device)
    groups, group_description = optimizer_groups(model, grouped)
    optimizer = torch.optim.AdamW(groups, lr=.001, weight_decay=.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    # A separate, fixed RNG ensures diagnostics never change training examples.
    diag_rng = torch.Generator().manual_seed(20260926)
    diag_starts = torch.randint(len(tokens)-257, (256,), generator=diag_rng)
    diagnostic_hash = hashlib.sha256(diag_starts.numpy().tobytes()).hexdigest()
    diag_batches = [tokens[starts.to(device)[:, None] + torch.arange(257, device=device)]
                    for starts in diag_starts.split(32)]
    sample_hash, lr_hash = hashlib.sha256(), hashlib.sha256()
    history, diagnostics = [], []
    train_seconds = diagnostic_seconds = checkpoint_seconds = 0.
    best_bpb, best_step = float('inf'), None

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def save_checkpoint(path, step):
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save({'protocol': PROTOCOL, 'implementation': implementation, 'config': config,
                    'model': state, 'seed': args.seed,
                    'train_tokens': step*args.batch_size*256}, path)

    def diagnostic(step):
        nonlocal best_bpb, best_step, diagnostic_seconds, checkpoint_seconds
        sync()
        started = time.perf_counter()
        train = training_diagnostic(model, diag_batches)
        validation = score(model, *data['validation'], device, 'fp32')
        validation.pop('window_nll_nats')
        validation['mean_token_nll'] = validation['nll_nats'] / validation['targets']
        sync()
        diagnostic_seconds += time.perf_counter() - started
        row = {'step': step, 'train_tokens': step*args.batch_size*256,
               'train_subset': train, 'validation': validation}
        diagnostics.append(row)
        print(json.dumps({'diagnostic': row}), flush=True)
        if step > 0:
            started = time.perf_counter()
            save_checkpoint(args.run_dir / f'checkpoint-step-{step:06d}.pt', step)
            if validation['bpb'] < best_bpb:
                best_bpb, best_step = validation['bpb'], step
            checkpoint_seconds += time.perf_counter() - started

    diagnostic(0)
    sync()
    segment_start = time.perf_counter()
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng)
        sample_hash.update(starts.numpy().tobytes())
        batch = tokens[starts.to(device)[:, None]+torch.arange(257, device=device)]
        lr = .001*min(1., (step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        lr_hash.update(struct.pack('<d', lr))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step+1) % 100 == 0 or step+1 == args.steps:
            row = {'step': step+1, 'loss': loss.item(), 'lr': lr}
            history.append(row)
            print(json.dumps(row), flush=True)
        if (step+1) % args.eval_every == 0 or step+1 == args.steps:
            sync()
            train_seconds += time.perf_counter()-segment_start
            diagnostic(step+1)
            sync()
            segment_start = time.perf_counter()
    final_path = args.run_dir / 'checkpoint.pt'
    save_checkpoint(final_path, args.steps)
    result = {
        'protocol': PROTOCOL, 'experiment': args.experiment, 'label': label,
        'implementation': implementation, 'config': config, 'seed': args.seed,
        'steps': args.steps, 'batch_size': args.batch_size, 'precision': precision,
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_tokens': args.steps*args.batch_size*256, 'parent_checkpoints': [],
        'train_seconds': train_seconds, 'diagnostic_seconds': diagnostic_seconds,
        'checkpoint_seconds': checkpoint_seconds,
        'process_seconds': time.perf_counter()-process_start,
        'validation': diagnostics[-1]['validation'], 'diagnostics': diagnostics, 'history': history,
        'best_validation_step': best_step, 'best_validation_bpb': best_bpb,
        'comparison_checkpoint': 'checkpoint.pt (fixed final step)',
        'training_window_sha256': sample_hash.hexdigest(), 'learning_rate_sha256': lr_hash.hexdigest(),
        'diagnostic_window_sha256': diagnostic_hash, 'diagnostic_seed': 20260926,
        'optimizer_groups': group_description, 'threads': args.threads,
        'torch_version': str(torch.__version__), 'checkpoint_sha256': sha(final_path),
        'implementation_sha256': implementation_sha,
        'source_sha256': {name: sha(ROOT/name) for name in (
            'train_round1.py', 'student.py', 'model.py', 'common.py', 'evaluate.py')},
        **device_metrics(device),
    }
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('experiment', 'seed', 'parameters', 'train_tokens',
                     'train_seconds', 'validation', 'best_validation_step')}), flush=True)


if __name__ == '__main__':
    main()
