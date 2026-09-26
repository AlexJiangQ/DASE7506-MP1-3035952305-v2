"""FP32 factorial-study trainer; immutable protocol, independent sampling, resumable updates."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, setup, sha, device_metrics
from evaluate import score
from experiment_utils import (WORK, write_json, append_event, check_disk, source_hashes,
                              load_development_data, rng_state, restore_rng, cpu_tree,
                              inference_checkpoint, peak_ram_bytes)
from student import build_seeded_model


class Engine:
    def __init__(self, config, seed=17, device='cpu', steps=1200, batch_size=32,
                 microbatch=32, lr=.001):
        if steps < 1 or batch_size < 1 or microbatch < 1 or batch_size % microbatch:
            raise ValueError('Positive steps and an integral number of microbatches required.')
        if config['context'] != 256 or config['vocab'] != 2048:
            raise ValueError('Fixed protocol requires context=256, vocab=2048.')
        self.config, self.seed, self.device = dict(config), seed, torch.device(device)
        self.steps, self.batch_size, self.microbatch, self.lr = steps, batch_size, microbatch, lr
        self.model = build_seeded_model(config, seed).to(self.device).train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=.1)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        random.seed(seed + 100000)
        np.random.seed(seed + 100000)
        torch.manual_seed(seed + 100000)
        self.update, self.targets = 0, 0
        self.history, self.validation_history = [], []
        self.best = None
        self.train_seconds = self.validation_seconds = self.save_seconds = 0.
        self.schedule_digest = '0' * 64

    def recipe(self):
        return {'config': self.config, 'seed': self.seed, 'steps': self.steps,
                'batch_size': self.batch_size, 'microbatch': self.microbatch, 'lr': self.lr}

    def step(self, tokens):
        if self.update >= self.steps:
            raise ValueError('Predeclared training endpoint reached.')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        # Keep the official randint upper bound, including its endpoint convention.
        starts = torch.randint(len(tokens) - 257, (self.batch_size,), generator=self.generator)
        indices = starts.to(self.device)
        batch = tokens[indices[:, None] + torch.arange(257, device=self.device)]
        rate = self.lr * min(1., (self.update + 1) / 100) * (
            .1 + .9 * .5 * (1 + math.cos(math.pi * self.update / self.steps)))
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for offset in range(0, self.batch_size, self.microbatch):
            mb = batch[offset:offset + self.microbatch]
            loss = F.cross_entropy(self.model(mb[:, :-1]).flatten(0, 1).float(), mb[:, 1:].flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            (loss * (len(mb) / self.batch_size)).backward()
            total_loss += loss.item() * len(mb) / self.batch_size
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.update += 1
        self.targets += self.batch_size * 256
        self.schedule_digest = hashlib.sha256(bytes.fromhex(self.schedule_digest) + starts.numpy().tobytes()).hexdigest()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        self.train_seconds += seconds
        return {'step': self.update, 'targets': self.targets, 'loss': total_loss, 'lr': rate,
                'grad_norm': float(norm), 'seconds': seconds, 'window_starts': starts.tolist(),
                'schedule_digest': self.schedule_digest}

    def validate(self, validation):
        started = time.perf_counter()
        # Do not migrate the live optimizer/model or disturb the training RNG streams.
        cpu_model = copy.deepcopy(self.model).cpu().float().eval()
        result = score(cpu_model, *validation, torch.device('cpu'), 'fp32')
        result.pop('window_nll_nats')
        result.update(step=self.update, train_targets=self.targets, device='cpu', precision='fp32')
        self.validation_history.append(result)
        self.validation_seconds += time.perf_counter() - started
        return result

    def snapshot(self):
        return {'protocol': PROTOCOL, 'recipe': self.recipe(), 'model': cpu_tree(self.model.state_dict()),
                'optimizer': cpu_tree(self.optimizer.state_dict()), 'rng': rng_state(self.generator),
                'update': self.update, 'targets': self.targets, 'history': self.history,
                'validation_history': self.validation_history, 'best': self.best,
                'train_seconds': self.train_seconds, 'validation_seconds': self.validation_seconds,
                'save_seconds': self.save_seconds, 'schedule_digest': self.schedule_digest,
                'scheduler': {'name': 'official_warmup_cosine', 'next_step': self.update,
                              'warmup': 100, 'total_steps': self.steps, 'peak_lr': self.lr},
                'source_hashes': source_hashes(), 'device_type': self.device.type}

    def restore(self, saved, verify_sources=True):
        if saved['protocol'] != PROTOCOL or saved['recipe'] != self.recipe():
            raise ValueError('Resume recipe/protocol does not match; no training was performed.')
        if verify_sources and saved['source_hashes'] != source_hashes():
            raise ValueError('Source changed since resume point; start a new experiment.')
        if saved['device_type'] != self.device.type:
            raise ValueError('Exact-resume runs must retain their device type.')
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])
        for key in ('update', 'targets', 'history', 'validation_history', 'best', 'train_seconds',
                    'validation_seconds', 'save_seconds', 'schedule_digest'):
            setattr(self, key, saved[key])
        if self.targets != self.update * self.batch_size * 256 or saved['scheduler']['next_step'] != self.update:
            raise ValueError('Inconsistent resume counters.')
        restore_rng(saved['rng'], self.generator)


def selection_eligible(update, endpoint):
    return update == 1200 if endpoint == 1200 else update >= 1200 and update % 600 == 0


def save_inference(engine, path):
    check_disk()
    implementation = ('model' if engine.config.get('position_encoding', 'learned') == 'learned'
                      and engine.config.get('ffn_type', 'gelu') == 'gelu' else 'student')
    torch.save(inference_checkpoint(engine.model, engine.config, engine.seed, engine.update,
                                   engine.batch_size, implementation), path)


def save_resume(engine, directory):
    check_disk()
    started = time.perf_counter()
    # Alternating slots retain the previous complete update if a write is interrupted.
    slot = directory / f'resume-{(engine.update // 100) % 2}.pt'
    torch.save(engine.snapshot(), slot)
    write_json(slot.with_suffix('.sha256.json'), {'sha256': sha(slot), 'step': engine.update})
    engine.save_seconds += time.perf_counter() - started


def latest_resume(directory):
    valid = []
    for slot in Path(directory).glob('resume-*.pt'):
        checksum = slot.with_suffix('.sha256.json')
        if checksum.exists():
            meta = json.loads(checksum.read_text())
            if meta['sha256'] == sha(slot):
                valid.append((meta['step'], slot))
    if not valid:
        raise ValueError('No complete checksum-verified resume point.')
    return max(valid)[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--microbatch', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=300)
    p.add_argument('--save-at', nargs='*', type=int, default=[])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Pause at a complete update, retaining the original LR horizon.')
    args = p.parse_args()
    if any(step < 1 or step > args.steps for step in args.save_at):
        p.error('--save-at steps must lie inside the declared training horizon.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        p.error('Run directory not empty; choose a new directory or explicitly resume.')
    check_disk()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    events = args.run_dir / 'events.jsonl'
    append_event(events, {'event': 'process_start', 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    try:
        device, _ = setup(args.device, 'fp32', args.threads)
        config = json.loads(args.config.read_text())
        data = load_development_data()
        engine = Engine(config, args.seed, device, args.steps, args.batch_size, args.microbatch, args.lr)
        if args.resume:
            resume_path = latest_resume(args.run_dir)
            engine.restore(torch.load(resume_path, map_location='cpu', weights_only=True))
            append_event(events, {'event': 'resume', 'path': str(resume_path), 'step': engine.update})
        tokens = data['train'][0].to(device)
        write_json(args.run_dir / 'recipe.json', engine.recipe() | {'source_hashes': source_hashes(),
            'torch': str(torch.__version__), 'threads': args.threads, 'precision': 'fp32',
            'selection': 'minimum CPU validation BPB at steps 1200,1800,...,T; ties earlier',
            'save_at': sorted(set(args.save_at)),
            'seeds': {'model': args.seed, 'data': args.seed, 'dropout': args.seed + 100000}})
        stop = min(args.steps, args.stop_after or args.steps)
        while engine.update < stop:
            row = engine.step(tokens)
            append_event(events, {'event': 'update', **row})
            if engine.update % 100 == 0 or engine.update == stop:
                engine.history.append({k: v for k, v in row.items() if k != 'window_starts'})
                print(json.dumps({'training': engine.history[-1]}), flush=True)
            if engine.update == args.steps or (args.eval_every > 0 and engine.update % args.eval_every == 0):
                result = engine.validate(data['validation'])
                append_event(events, {'event': 'validation', **result})
                print(json.dumps({'validation': result}), flush=True)
                if selection_eligible(engine.update, args.steps) and (engine.best is None or result['bpb'] < engine.best['bpb']):
                    engine.best = {'step': engine.update, 'bpb': result['bpb'], 'train_targets': engine.targets}
                    save_inference(engine, args.run_dir / 'selected.pt')
            if engine.update % 100 == 0 or engine.update == stop:
                save_resume(engine, args.run_dir)
            if engine.update in args.save_at:
                save_inference(engine, args.run_dir / f'checkpoint-step{engine.update}.pt')
        complete = engine.update == args.steps
        if complete:
            save_inference(engine, args.run_dir / 'checkpoint.pt')
        metrics = {'protocol': PROTOCOL, **engine.recipe(), 'complete': complete, 'updates': engine.update,
                   'train_targets': engine.targets, 'parameters': sum(p.numel() for p in engine.model.parameters()),
                   'train_seconds': engine.train_seconds, 'validation_seconds': engine.validation_seconds,
                   'save_seconds': engine.save_seconds, 'process_seconds_this_session': time.perf_counter() - started,
                   'peak_process_ram_bytes': peak_ram_bytes(), 'validation_history': engine.validation_history,
                   'selected': engine.best, 'schedule_digest': engine.schedule_digest,
                   'source_hashes': source_hashes(), **device_metrics(device)}
        if complete:
            metrics['endpoint_validation'] = engine.validation_history[-1]
            metrics['checkpoint_sha256'] = sha(args.run_dir / 'checkpoint.pt')
        write_json(args.run_dir / 'metrics.json', metrics)
        append_event(events, {'event': 'process_end', 'complete': complete, 'step': engine.update,
                              'seconds': time.perf_counter() - started})
        print(json.dumps({'complete': complete, 'run_dir': str(args.run_dir), 'updates': engine.update}), flush=True)
    except BaseException as error:
        append_event(events, {'event': 'failure', 'error': repr(error), 'traceback': traceback.format_exc(),
                              'seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
