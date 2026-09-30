"""Round-two research entry: one candidate change at a time against the frozen E3 recipe.

The loop, sampling stream, diagnostic grid and checkpoint format match
train_round1.py. ``learning_rate`` keeps the round-one expression for the default
cosine floor, so candidates that do not test the floor reproduce the recorded
learning-rate hash byte for byte rather than merely closely.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, device_metrics, load_data, make_model, setup, sha
from evaluate import score
from train_round1 import optimizer_groups, training_diagnostic

E3_CONFIG = {'vocab': 2048, 'width': 128, 'heads': 4, 'depth': 4, 'context': 256,
             'ffn': 'swiglu', 'ffn_hidden': 344}

# Cosine floor -> cosine amplitude. The two sum to one, so the peak stays at 1.0.
# Writing both as separate literals keeps that invariant visible and checkable
# rather than implied by arithmetic; it is documentation, not a numerical
# requirement, since (1 - .1) == .9 holds exactly in double precision. The default
# floor must reproduce the round-one learning-rate sequence exactly, and the test
# suite pins that stream to the recorded hash.
LR_FLOORS = {.1: .9, .2: .8}

# The two structures the overnight queue compares, and the only two it may vary.
# CAP_CONFIG is the scaled shape the eval-budget probe showed still fits the 5x CPU
# limit; P3A_CONFIG is the current best. Sharing the constant between the 4,000-step
# and 30,000-step entry of a structure makes "steps are the only difference"
# structural rather than a claim about two hand-copied dicts.
CAP_CONFIG = {'vocab': 2048, 'width': 256, 'heads': 8, 'depth': 6, 'context': 256,
              'ffn': 'swiglu', 'ffn_hidden': 688, 'pos': 'rope'}
P3A_CONFIG = E3_CONFIG | {'width': 192, 'heads': 6, 'ffn_hidden': 512, 'pos': 'rope'}

CANDIDATES = {
    'C-depth': dict(
        implementation='student', lr_floor=.1, config=E3_CONFIG | {'depth': 6},
        question='Does six layers instead of four improve BPB at the same target count?'),
    'C-width': dict(
        implementation='student', lr_floor=.1,
        config=E3_CONFIG | {'width': 192, 'heads': 6, 'ffn_hidden': 512},
        question='Does width 192 with six heads improve BPB at the same target count?'),
    'R-rope': dict(
        implementation='research_models', lr_floor=.1, config=E3_CONFIG | {'pos': 'rope'},
        question='Does rotary position encoding beat the learned absolute embedding?'),
    'T-lrfloor': dict(
        implementation='student', lr_floor=.2, config=dict(E3_CONFIG),
        question='Does raising the cosine floor from 10% to 20% of the peak help?'),
    'P3A': dict(
        implementation='research_models', lr_floor=.1, config=P3A_CONFIG,
        question='Does RoPE on top of the C-width capacity lower BPB further?'),
    # Overnight capacity queue: {2.17M, 5.29M} x {4000, 30000} steps, plus a dropout
    # pair at the larger size. Steps come from --steps, so an entry names a structure
    # and a budget; dropout is the only config difference inside the pair.
    'CAP-4k': dict(
        implementation='research_models', lr_floor=.1, config=CAP_CONFIG,
        question='Does the scaled structure at 4,000 steps beat P3A at the same token count?'),
    'CAP-30k-p0': dict(
        implementation='research_models', lr_floor=.1, config=CAP_CONFIG,
        question='How far does the scaled structure go when the 30,000-step budget is spent on it?'),
    'CAP-30k-p01': dict(
        implementation='research_models', lr_floor=.1, config=CAP_CONFIG | {'dropout': .1},
        question='Does residual dropout 0.1 help at 46 tokens per parameter?'),
    'P3A-30k': dict(
        implementation='research_models', lr_floor=.1, config=P3A_CONFIG,
        question='How much of the capacity gain is only the longer schedule?'),
    'CAP-p0': dict(
        implementation='research_models', lr_floor=.1, config=CAP_CONFIG,
        question='What does the scaled architecture achieve without dropout at a matched medium budget?'),
    'CAP-p02': dict(
        implementation='research_models', lr_floor=.1,
        config=CAP_CONFIG | {'dropout': .2},
        question='Can residual dropout 0.2 preserve the scaled model generalization over a medium budget?'),
    'CAP-p03': dict(
        implementation='research_models', lr_floor=.1,
        config=CAP_CONFIG | {'dropout': .3},
        question='Does residual dropout 0.3 beat 0.2 on the recipe that gave the lowest validation BPB?'),
}


def learning_rate(step, steps, floor=.1, peak=.001):
    """100-step linear warmup, then cosine decay from the peak to ``floor`` of the peak.

    ``peak`` is the warmed-up learning rate. It defaults to the round-one value, so
    the default call reproduces the recorded learning-rate stream byte for byte;
    scaling it only rescales the whole schedule and leaves its shape untouched.
    """
    return peak*min(1., (step+1)/100)*(floor+LR_FLOORS[floor]*.5*(1+math.cos(math.pi*step/steps)))


def rebuild_prefix_hashes(train_length, batch_size, seed, completed_steps, steps, floor, peak):
    """Recreate the already-consumed sampler and LR streams without training."""
    sampler = torch.Generator().manual_seed(seed)
    sample_hash, lr_hash = hashlib.sha256(), hashlib.sha256()
    for step in range(completed_steps):
        starts = torch.randint(train_length-257, (batch_size,), generator=sampler)
        sample_hash.update(starts.numpy().tobytes())
        lr_hash.update(struct.pack('<d', learning_rate(step, steps, floor, peak)))
    return sampler, sample_hash, lr_hash


def atomic_save_resume(path, state):
    """Keep the preceding complete state if serialization is interrupted."""
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=CANDIDATES, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--peak-lr', type=float, default=.001)
    parser.add_argument('--adam-beta2', type=float, default=.999)
    parser.add_argument('--save-resume-state', action='store_true',
                        help='Atomically save a complete training state after every diagnostic')
    parser.add_argument('--resume-state', type=Path,
                        help='Continue the same run from a complete state; also keeps updating that state')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1:
        parser.error('steps, batch-size and eval-every must be positive')
    if not args.peak_lr > 0:
        parser.error('peak-lr must be positive')
    if not 0 < args.adam_beta2 < 1:
        parser.error('adam-beta2 must be between 0 and 1')
    if args.resume_state and not args.resume_state.is_file():
        parser.error(f'Resume state does not exist: {args.resume_state}')
    if args.resume_state and not args.run_dir.is_dir():
        parser.error('The original run directory must exist when resuming')
    if not args.resume_state and args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Use a fresh output directory')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume_state or (args.run_dir/'resume-state.pt'
                                        if args.save_resume_state else None)
    process_start = time.perf_counter()
    device, precision = setup(args.device, 'fp32', args.threads)
    torch.manual_seed(args.seed)
    data = load_data()
    candidate = CANDIDATES[args.candidate]
    implementation, floor = candidate['implementation'], candidate['lr_floor']
    config = dict(candidate['config'])
    model, implementation_sha = make_model(implementation, config, device)
    # Every candidate inherits the round-one E3 training treatment: grouped decay.
    groups, group_description = optimizer_groups(model, True)
    optimizer = torch.optim.AdamW(groups, lr=.001, weight_decay=.1,
                                  betas=(.9, args.adam_beta2))
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
    start_step, process_seconds_prior = 0, 0.
    resume_metadata = {
        'protocol': PROTOCOL, 'candidate': args.candidate,
        'implementation': implementation, 'implementation_sha256': implementation_sha,
        'train_source_sha256': sha(ROOT/'train_research.py'),
        'data_manifest_sha256': sha(ROOT/'data/manifest.json'),
        'config': config, 'seed': args.seed, 'steps': args.steps,
        'batch_size': args.batch_size, 'eval_every': args.eval_every,
        'peak_lr': args.peak_lr, 'lr_floor': floor, 'adam_beta2': args.adam_beta2,
        'device': str(device), 'device_name': torch.cuda.get_device_name(device)
        if device.type == 'cuda' else 'CPU',
        'threads': args.threads, 'torch_version': str(torch.__version__),
        'run_dir': str(args.run_dir.resolve()),
    }
    if args.resume_state:
        saved = torch.load(args.resume_state, map_location='cpu', weights_only=False)
        if saved.get('version') != 1:
            raise ValueError('Unsupported resume-state version')
        for key, expected in resume_metadata.items():
            if saved.get('metadata', {}).get(key) != expected:
                raise ValueError(f'Resume state mismatch for {key}: '
                                 f'{saved.get("metadata", {}).get(key)!r} != {expected!r}')
        start_step = saved['step']
        if (not isinstance(start_step, int) or start_step < 0 or start_step > args.steps
                or (start_step % args.eval_every and start_step != args.steps)):
            raise ValueError(f'Invalid saved diagnostic step: {start_step!r}')
        if not saved['diagnostics'] or saved['diagnostics'][-1]['step'] != start_step:
            raise ValueError('Resume diagnostics do not end at the saved step')
        rebuilt_rng, sample_hash, lr_hash = rebuild_prefix_hashes(
            len(tokens), args.batch_size, args.seed, start_step, args.steps, floor, args.peak_lr)
        if (sample_hash.hexdigest() != saved['training_window_sha256']
                or lr_hash.hexdigest() != saved['learning_rate_sha256']
                or not torch.equal(rebuilt_rng.get_state(), saved['sampler_rng_state'])):
            raise ValueError('Resume sampler or learning-rate prefix does not match')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng_state'])
        if device.type == 'cuda':
            if len(saved['cuda_rng_states']) != torch.cuda.device_count():
                raise ValueError('CUDA device count differs from saved resume state')
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        rng.set_state(saved['sampler_rng_state'])
        history, diagnostics = saved['history'], saved['diagnostics']
        best_bpb, best_step = saved['best_bpb'], saved['best_step']
        train_seconds = saved['train_seconds']
        diagnostic_seconds = saved['diagnostic_seconds']
        checkpoint_seconds = saved['checkpoint_seconds']
        process_seconds_prior = saved['process_seconds']
        print(f'Resumed {args.candidate} at step {start_step} from {args.resume_state}', flush=True)

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
        if resume_path is not None:
            atomic_save_resume(resume_path, {
                'version': 1, 'metadata': resume_metadata, 'step': step,
                'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                'model_training': model.training,
                'optimizer': optimizer.state_dict(),
                'cpu_rng_state': torch.get_rng_state(),
                'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                'sampler_rng_state': rng.get_state(),
                'training_window_sha256': sample_hash.hexdigest(),
                'learning_rate_sha256': lr_hash.hexdigest(),
                'history': history, 'diagnostics': diagnostics,
                'best_bpb': best_bpb, 'best_step': best_step,
                'train_seconds': train_seconds, 'diagnostic_seconds': diagnostic_seconds,
                'checkpoint_seconds': checkpoint_seconds,
                'process_seconds': process_seconds_prior + time.perf_counter()-process_start,
            })

    if args.resume_state:
        model.train(saved['model_training'])
    else:
        diagnostic(0)
    sync()
    segment_start = time.perf_counter()
    for step in range(start_step, args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng)
        sample_hash.update(starts.numpy().tobytes())
        batch = tokens[starts.to(device)[:, None]+torch.arange(257, device=device)]
        lr = learning_rate(step, args.steps, floor, args.peak_lr)
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
        'protocol': PROTOCOL, 'candidate': args.candidate, 'question': candidate['question'],
        'implementation': implementation, 'config': config, 'seed': args.seed,
        'steps': args.steps, 'batch_size': args.batch_size, 'precision': precision,
        'lr_floor': floor, 'lr_cosine_amplitude': LR_FLOORS[floor], 'peak_lr': args.peak_lr,
        'adam_betas': [.9, args.adam_beta2],
        'weight_decay': {'matrices_and_embeddings': .1, 'layernorm_and_biases': 0.},
        'anchor': (f'E3 seed {args.seed} at the same 4000-step budget'
                   if args.steps == 4000 else
                   f'E3 seed {args.seed} historical 4000-step reference; '
                   f'not a same-budget control for {args.steps} steps'),
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_tokens': args.steps*args.batch_size*256, 'parent_checkpoints': [],
        'train_seconds': train_seconds, 'diagnostic_seconds': diagnostic_seconds,
        'checkpoint_seconds': checkpoint_seconds,
        'process_seconds': process_seconds_prior + time.perf_counter()-process_start,
        'validation': diagnostics[-1]['validation'], 'diagnostics': diagnostics, 'history': history,
        'best_validation_step': best_step, 'best_validation_bpb': best_bpb,
        'comparison_checkpoint': 'checkpoint.pt (fixed final step)',
        'training_window_sha256': sample_hash.hexdigest(), 'learning_rate_sha256': lr_hash.hexdigest(),
        'diagnostic_window_sha256': diagnostic_hash, 'diagnostic_seed': 20260926,
        'optimizer_groups': group_description, 'threads': args.threads,
        'torch_version': str(torch.__version__), 'checkpoint_sha256': sha(final_path),
        'implementation_sha256': implementation_sha,
        'source_sha256': {name: sha(ROOT/name) for name in (
            'train_research.py', 'research_models.py', 'train_round1.py', 'student.py',
            'model.py', 'common.py', 'evaluate.py')},
        **device_metrics(device),
    }
    (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('candidate', 'seed', 'parameters', 'train_tokens',
                     'train_seconds', 'validation', 'best_validation_step')}), flush=True)


if __name__ == '__main__':
    main()
