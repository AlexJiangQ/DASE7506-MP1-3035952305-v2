"""Train a randomly initialized student from train-only hard and teacher targets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import torch
from torch.nn import functional as F
from common import PROTOCOL, make_model, setup, sha, device_metrics
from experiment_utils import (append_event, check_disk, load_development_data,
                              peak_ram_bytes, source_hashes, write_json)
from train_experiment import (Engine, latest_resume, save_inference, save_resume,
                              selection_eligible)


def mixed_teacher_probabilities(logits, temperature):
    if not logits:
        raise ValueError('At least one teacher is required.')
    probabilities = [F.softmax(value.float() / temperature, dim=-1) for value in logits]
    return torch.stack(probabilities).mean(0)


def distillation_loss(student_logits, targets, teacher_logits, weight, temperature):
    hard = F.cross_entropy(student_logits.flatten(0, 1).float(), targets.flatten())
    teacher_p = mixed_teacher_probabilities(teacher_logits, temperature)
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    soft = F.kl_div(student_logp, teacher_p, reduction='none').sum(-1).mean() * temperature ** 2
    total = (1. - weight) * hard + weight * soft
    return total, hard.detach(), soft.detach()


def linear_distill_weight(start, end, update, steps):
    if steps < 1:
        raise ValueError('Training horizon must be positive.')
    if update < 0 or update >= steps:
        raise ValueError('Update must lie inside the training horizon.')
    progress = 1. if steps == 1 else update / (steps - 1)
    return float(start + (end - start) * progress)


def load_teacher(path, device):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError(f'Teacher protocol mismatch: {path}')
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval().requires_grad_(False)
    ancestry = {'path': str(path.resolve()), 'sha256': sha(path),
                'implementation': checkpoint['implementation'], 'config': checkpoint['config'],
                'seed': checkpoint.get('seed'), 'updates': checkpoint.get('updates'),
                'train_tokens': checkpoint.get('train_tokens')}
    return model, ancestry


class DistillEngine(Engine):
    def __init__(self, config, teacher_paths, distill_weight, temperature,
                 distill_weight_end=None, **kwargs):
        distill_weight_end = distill_weight if distill_weight_end is None else distill_weight_end
        if not 0. < distill_weight < 1. or not 0. < distill_weight_end < 1.:
            raise ValueError('Distillation weights must lie strictly between zero and one.')
        if temperature <= 0.:
            raise ValueError('Temperature must be positive.')
        super().__init__(config, **kwargs)
        loaded = [load_teacher(path, self.device) for path in teacher_paths]
        self.teachers = [item[0] for item in loaded]
        self.ancestry = [item[1] for item in loaded]
        self.distill_weight = float(distill_weight)
        self.distill_weight_end = float(distill_weight_end)
        self.temperature = float(temperature)

    def current_distill_weight(self):
        return linear_distill_weight(
            self.distill_weight, self.distill_weight_end, self.update, self.steps)

    def recipe(self):
        recipe = super().recipe()
        recipe['distillation'] = {'weight': self.distill_weight,
                                  'weight_start': self.distill_weight,
                                  'weight_end': self.distill_weight_end,
                                  'weight_schedule': 'linear_by_optimizer_update',
                                  'temperature': self.temperature,
                                  'teacher_sha256': [item['sha256'] for item in self.ancestry],
                                  'teacher_mixture': 'arithmetic_mean_probabilities'}
        return recipe

    def step(self, tokens):
        if self.update >= self.steps:
            raise ValueError('Predeclared training endpoint reached.')
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        starts = torch.randint(len(tokens) - 257, (self.batch_size,), generator=self.generator)
        indices = starts.to(self.device)
        batch = tokens[indices[:, None] + torch.arange(257, device=self.device)]
        rate = self.lr * min(1., (self.update + 1) / 100) * (
            .1 + .9 * .5 * (1 + math.cos(math.pi * self.update / self.steps)))
        for group in self.optimizer.param_groups:
            group['lr'] = rate
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = hard_loss = soft_loss = 0.
        distill_weight = self.current_distill_weight()
        for offset in range(0, self.batch_size, self.microbatch):
            mb = batch[offset:offset + self.microbatch]
            with torch.no_grad():
                teacher_logits = [teacher(mb[:, :-1]) for teacher in self.teachers]
            student_logits = self.model(mb[:, :-1])
            loss, hard, soft = distillation_loss(
                student_logits, mb[:, 1:], teacher_logits,
                distill_weight, self.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite distillation loss.')
            fraction = len(mb) / self.batch_size
            (loss * fraction).backward()
            total_loss += loss.item() * fraction
            hard_loss += hard.item() * fraction
            soft_loss += soft.item() * fraction
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.update += 1
        self.targets += self.batch_size * 256
        self.schedule_digest = hashlib.sha256(
            bytes.fromhex(self.schedule_digest) + starts.numpy().tobytes()).hexdigest()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        self.train_seconds += seconds
        return {'step': self.update, 'targets': self.targets, 'loss': total_loss,
                'hard_loss': hard_loss, 'soft_loss': soft_loss, 'lr': rate,
                'distill_weight': distill_weight,
                'grad_norm': float(norm), 'seconds': seconds,
                'window_starts': starts.tolist(), 'schedule_digest': self.schedule_digest}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--teacher', action='append', type=Path, required=True)
    p.add_argument('--distill-weight', type=float, required=True)
    p.add_argument('--distill-weight-end', type=float)
    p.add_argument('--temperature', type=float, default=2.)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=3600)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--microbatch', type=int, default=32)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=600)
    p.add_argument('--save-at', nargs='*', type=int, default=[])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    if any(step < 1 or step > args.steps for step in args.save_at):
        p.error('--save-at steps must lie inside the declared training horizon.')
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        p.error('Run directory not empty; choose a new directory or explicitly resume.')
    check_disk()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    events = args.run_dir / 'events.jsonl'
    append_event(events, {'event': 'process_start', 'arguments': {
        key: ([str(x) for x in value] if key == 'teacher' else str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()}})
    try:
        device, _ = setup(args.device, 'fp32', args.threads)
        config = json.loads(args.config.read_text())
        data = load_development_data()
        engine = DistillEngine(config, args.teacher, args.distill_weight, args.temperature,
                               distill_weight_end=args.distill_weight_end,
                               seed=args.seed, device=device, steps=args.steps,
                               batch_size=args.batch_size, microbatch=args.microbatch, lr=args.lr)
        if args.resume:
            resume_path = latest_resume(args.run_dir)
            engine.restore(torch.load(resume_path, map_location='cpu', weights_only=True))
            append_event(events, {'event': 'resume', 'path': str(resume_path), 'step': engine.update})
        tokens = data['train'][0].to(device)
        write_json(args.run_dir / 'ancestry.json', {'student_config': config,
                   'student_seed': args.seed, 'teachers': engine.ancestry})
        write_json(args.run_dir / 'recipe.json', engine.recipe() | {
            'source_hashes': source_hashes(), 'torch': str(torch.__version__),
            'threads': args.threads, 'precision': 'fp32',
            'selection': 'minimum CPU validation BPB at steps 1200,1800,...,T; ties earlier',
            'save_at': sorted(set(args.save_at)),
            'data_policy': 'train-only gradients and teacher queries; validation scoring only'})
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
                if selection_eligible(engine.update, args.steps) and (
                        engine.best is None or result['bpb'] < engine.best['bpb']):
                    engine.best = {'step': engine.update, 'bpb': result['bpb'],
                                   'train_targets': engine.targets}
                    save_inference(engine, args.run_dir / 'selected.pt')
            if engine.update % 100 == 0 or engine.update == stop:
                save_resume(engine, args.run_dir)
            if engine.update in args.save_at:
                save_inference(engine, args.run_dir / f'checkpoint-step{engine.update}.pt')
        complete = engine.update == args.steps
        if complete:
            save_inference(engine, args.run_dir / 'checkpoint.pt')
        metrics = {'protocol': PROTOCOL, **engine.recipe(), 'complete': complete,
                   'updates': engine.update, 'train_targets': engine.targets,
                   'parameters': sum(p.numel() for p in engine.model.parameters()),
                   'teacher_parameters': [sum(p.numel() for p in t.parameters()) for t in engine.teachers],
                   'train_seconds': engine.train_seconds,
                   'validation_seconds': engine.validation_seconds,
                   'save_seconds': engine.save_seconds,
                   'process_seconds_this_session': time.perf_counter() - started,
                   'peak_process_ram_bytes': peak_ram_bytes(),
                   'validation_history': engine.validation_history,
                   'selected': engine.best, 'schedule_digest': engine.schedule_digest,
                   'source_hashes': source_hashes(), 'ancestry': engine.ancestry,
                   **device_metrics(device)}
        if complete:
            metrics['endpoint_validation'] = engine.validation_history[-1]
            metrics['checkpoint_sha256'] = sha(args.run_dir / 'checkpoint.pt')
        write_json(args.run_dir / 'metrics.json', metrics)
        append_event(events, {'event': 'process_end', 'complete': complete,
                              'step': engine.update, 'seconds': time.perf_counter() - started})
        print(json.dumps({'complete': complete, 'run_dir': str(args.run_dir),
                          'updates': engine.update}), flush=True)
    except BaseException as error:
        append_event(events, {'event': 'failure', 'error': repr(error),
                              'traceback': traceback.format_exc(),
                              'seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
