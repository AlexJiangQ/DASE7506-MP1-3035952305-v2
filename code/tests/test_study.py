"""Study-specific correctness checks; tiny random inputs, no test-split access."""
import copy
import io
import math
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from common import windows, PROTOCOL, make_model
from evaluate import score
from model import GPT
from student import build_model, build_seeded_model, swiglu_width
from train_experiment import Engine, selection_eligible
from experiment_utils import inference_checkpoint
from distill_experiment import (DistillEngine, distillation_loss,
                                linear_distill_weight, mixed_teacher_probabilities)
from checkpoint_average import average_checkpoints


def config(group='A', width=32, depth=2):
    return dict(vocab=2048, context=256, width=width, depth=depth, heads=4,
                position_encoding='rope' if group in 'BD' else 'learned',
                ffn_type='swiglu' if group in 'CD' else 'gelu')


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')

    def test_all_contracts_all_variants_and_lengths(self):
        for group in 'ABCD':
            model = build_seeded_model(config(group), 17).eval()
            for length in (1, 7, 256):
                with self.subTest(group=group, length=length), torch.no_grad():
                    x = torch.randint(2048, (2, length))
                    a = model.predict_log_probs(x)
                    self.assertEqual(tuple(a.shape), (2, length, 2048))
                    self.assertTrue(torch.isfinite(a).all())
                    torch.testing.assert_close(a.logsumexp(-1), torch.zeros(2, length), atol=1e-6, rtol=1e-6)
                    torch.testing.assert_close(a[:1], model.predict_log_probs(x[:1]), atol=1e-5, rtol=1e-5)
                    model.predict_log_probs((x + 9) % 2048)
                    self.assertTrue(torch.equal(a, model.predict_log_probs(x)))
                    cutoff = max(1, length // 2)
                    changed = x.clone()
                    changed[:, cutoff:] = (changed[:, cutoff:] + 19) % 2048
                    torch.testing.assert_close(a[:, :cutoff], model.predict_log_probs(changed)[:, :cutoff], atol=1e-6, rtol=1e-6)

    def test_all_parameter_gradients_are_finite_and_nonzero(self):
        for group in 'ABCD':
            with self.subTest(group=group):
                model = build_seeded_model(config(group), 17)
                ids = torch.randint(2048, (2, 14))
                F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten()).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_shared_initialization_and_weight_tying(self):
        for seed in (17, 42, 123):
            models = [build_seeded_model(config(g), seed) for g in 'ABCD']
            base = models[0].state_dict()
            for model in models:
                self.assertIs(model.head.weight, model.token.weight)
                for name, value in model.state_dict().items():
                    if name in base and value.shape == base[name].shape:
                        self.assertTrue(torch.equal(value, base[name]), name)
            for name, value in models[2].state_dict().items():
                if '.mlp.' in name:
                    self.assertTrue(torch.equal(value, models[3].state_dict()[name]), name)

    def test_seed_factory_leaves_cpu_rng_unchanged(self):
        before = torch.get_rng_state().clone()
        build_seeded_model(config('D'), 17)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_no_change_model_exactly_matches_official(self):
        torch.manual_seed(17)
        original = GPT(config('A'))
        ours = build_seeded_model(config('A'), 17)
        ids = torch.randint(2048, (2, 23))
        self.assertTrue(torch.equal(original(ids), ours(ids)))

    def test_parameter_counts_and_rope_geometry(self):
        models = {g: build_seeded_model(config(g, 128, 4), 17) for g in 'ABCD'}
        counts = {g: sum(p.numel() for p in m.parameters()) for g, m in models.items()}
        self.assertEqual(counts['A'], 1088256)
        self.assertEqual(counts['A'] - counts['B'], 32768)
        self.assertEqual(counts['C'] - counts['D'], 32768)
        self.assertEqual(swiglu_width(128), 344)
        self.assertEqual(swiglu_width(192), 512)
        block = models['B'].blocks[0]
        x = torch.randn(1, 4, 256, 32)
        rotated = block.rotate(x)
        self.assertTrue(torch.equal(x[:, :, 0], rotated[:, :, 0]))
        torch.testing.assert_close(x.square().sum(-1), rotated.square().sum(-1), atol=1e-5, rtol=1e-6)

    def test_sampler_sequence_shared_across_models(self):
        tokens = torch.randint(2048, (1100,))
        sequences = []
        for group in 'ABCD':
            engine = Engine(config(group), steps=3, batch_size=2, microbatch=2)
            sequences.append([engine.step(tokens)['window_starts'] for _ in range(3)])
        self.assertTrue(all(s == sequences[0] for s in sequences))

    def test_default_recipe_short_training_matches_original(self):
        cfg = config('A')
        tokens = torch.randint(2048, (1100,))
        engine = Engine(cfg, steps=3, batch_size=2, microbatch=2)
        torch.manual_seed(17)
        official = GPT(cfg)
        optimizer = torch.optim.AdamW(official.parameters(), lr=.001, weight_decay=.1)
        sampler = torch.Generator().manual_seed(17)
        for step in range(3):
            row = engine.step(tokens)
            starts = torch.randint(len(tokens) - 257, (2,), generator=sampler)
            self.assertEqual(starts.tolist(), row['window_starts'])
            batch = tokens[starts[:, None] + torch.arange(257)]
            lr = .001 * min(1., (step + 1) / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / 3)))
            self.assertEqual(lr, row['lr'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(official(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(official.parameters(), 1.)
            optimizer.step()
            for name, value in official.state_dict().items():
                self.assertTrue(torch.equal(value, engine.model.state_dict()[name]), name)

    def assert_resume(self, device):
        tokens = torch.randint(2048, (1100,)).to(device)
        for group in 'ABCD':
            with self.subTest(group=group, device=device):
                uninterrupted = Engine(config(group), device=device, steps=6, batch_size=2, microbatch=2)
                rows = [uninterrupted.step(tokens) for _ in range(6)]
                interrupted = Engine(config(group), device=device, steps=6, batch_size=2, microbatch=2)
                for _ in range(3):
                    interrupted.step(tokens)
                stream = io.BytesIO()
                torch.save(interrupted.snapshot(), stream)
                stream.seek(0)
                restored = Engine(config(group), device=device, steps=6, batch_size=2, microbatch=2)
                restored.restore(torch.load(stream, map_location='cpu', weights_only=True))
                restored_rows = [restored.step(tokens) for _ in range(3)]
                for a, b in zip(rows[3:], restored_rows):
                    for field in ('window_starts', 'lr', 'step', 'targets', 'schedule_digest'):
                        self.assertEqual(a[field], b[field])
                max_error = 0.
                for name, value in uninterrupted.model.state_dict().items():
                    other = restored.model.state_dict()[name]
                    max_error = max(max_error, float((value - other).abs().max()))
                    if device == 'cpu':
                        self.assertTrue(torch.equal(value, other), name)
                    else:
                        torch.testing.assert_close(value, other, atol=1e-6, rtol=1e-5)
                print(f'resume {device} {group}: max_abs_error={max_error}', flush=True)

    def test_cpu_resume_is_bitwise_exact(self):
        self.assert_resume('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_resume_same_environment(self):
        self.assert_resume('cuda')

    def test_gradient_accumulation_matches_full_batch(self):
        tokens = torch.randint(2048, (1100,))
        full = Engine(config('D'), steps=1, batch_size=4, microbatch=4)
        accumulated = Engine(config('D'), steps=1, batch_size=4, microbatch=2)
        a, b = full.step(tokens), accumulated.step(tokens)
        self.assertEqual(a['window_starts'], b['window_starts'])
        self.assertAlmostEqual(a['loss'], b['loss'], places=5)
        for name, value in full.model.state_dict().items():
            torch.testing.assert_close(value, accumulated.model.state_dict()[name], atol=2e-6, rtol=1e-4)

    def test_official_loading_and_partial_final_window(self):
        tokens = torch.randint(2048, (2 * 256 + 7,))
        for group in 'ABCD':
            with self.subTest(group=group):
                model = build_seeded_model(config(group), 17)
                ckpt = inference_checkpoint(model, config(group), 17, 0, 32)
                stream = io.BytesIO()
                torch.save(ckpt, stream)
                stream.seek(0)
                loaded = torch.load(stream, weights_only=True)
                restored, _ = make_model(loaded['implementation'], loaded['config'], torch.device('cpu'))
                restored.load_state_dict(loaded['model'])
                self.assertIs(restored.token.weight, restored.head.weight)
                a = score(model, tokens, 2000, torch.device('cpu'), 'fp32', batch_size=2)
                b = score(restored, tokens, 2000, torch.device('cpu'), 'fp32', batch_size=2)
                self.assertEqual(a['targets'], len(tokens) - 1)
                self.assertEqual(a['bpb'], b['bpb'])

    def test_preregistered_checkpoint_grid(self):
        self.assertEqual([i for i in range(1, 1201) if selection_eligible(i, 1200)], [1200])
        self.assertEqual([i for i in range(1, 3601) if selection_eligible(i, 3600)], [1200, 1800, 2400, 3000, 3600])

    def test_distillation_probability_mixture_and_loss(self):
        a = torch.tensor([[[2., 0., -1.]]])
        b = torch.tensor([[[0., 2., -1.]]])
        mixed = mixed_teacher_probabilities([a, b], 2.)
        expected = (F.softmax(a / 2., -1) + F.softmax(b / 2., -1)) / 2
        torch.testing.assert_close(mixed, expected)
        student = torch.tensor([[[.2, -.1, .4]]], requires_grad=True)
        total, hard, soft = distillation_loss(student, torch.tensor([[2]]), [a, b], .3, 2.)
        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(total.item(), .7 * hard.item() + .3 * soft.item(), places=6)
        total.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_linear_distillation_weight_schedule(self):
        self.assertEqual(linear_distill_weight(.5, .2, 0, 4), .5)
        self.assertAlmostEqual(linear_distill_weight(.5, .2, 1, 4), .4)
        self.assertAlmostEqual(linear_distill_weight(.5, .2, 2, 4), .3)
        self.assertEqual(linear_distill_weight(.5, .2, 3, 4), .2)
        self.assertEqual(linear_distill_weight(.3, .3, 2, 4), .3)
        with self.assertRaises(ValueError):
            linear_distill_weight(.5, .2, 4, 4)

    def teacher_checkpoint(self, path, seed=91):
        cfg = config('D')
        model = build_seeded_model(cfg, seed)
        torch.save(inference_checkpoint(model, cfg, seed, 3, 2), path)
        return cfg

    def test_distill_engine_freezes_teacher_and_exports_student_only(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher_path = Path(directory) / 'teacher.pt'
            cfg = self.teacher_checkpoint(teacher_path)
            engine = DistillEngine(cfg, [teacher_path], .3, 2., steps=1,
                                   batch_size=2, microbatch=1)
            tokens = torch.randint(2048, (1100,))
            row = engine.step(tokens)
            self.assertTrue(math.isfinite(row['hard_loss']))
            self.assertTrue(math.isfinite(row['soft_loss']))
            self.assertEqual(row['distill_weight'], .3)
            self.assertTrue(all(parameter.grad is None for parameter in engine.teachers[0].parameters()))
            checkpoint = inference_checkpoint(engine.model, cfg, 17, 1, 2)
            self.assertNotIn('teacher', checkpoint)
            restored, _ = make_model(checkpoint['implementation'], checkpoint['config'], torch.device('cpu'))
            restored.load_state_dict(checkpoint['model'])

    def test_distill_resume_is_bitwise_exact_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            teacher_path = Path(directory) / 'teacher.pt'
            cfg = self.teacher_checkpoint(teacher_path)
            tokens = torch.randint(2048, (1100,))
            full = DistillEngine(cfg, [teacher_path], .3, 2., steps=3,
                                 batch_size=2, microbatch=2)
            full_rows = [full.step(tokens) for _ in range(3)]
            interrupted = DistillEngine(cfg, [teacher_path], .3, 2., steps=3,
                                        batch_size=2, microbatch=2)
            interrupted.step(tokens)
            stream = io.BytesIO()
            torch.save(interrupted.snapshot(), stream)
            stream.seek(0)
            restored = DistillEngine(cfg, [teacher_path], .3, 2., steps=3,
                                     batch_size=2, microbatch=2)
            restored.restore(torch.load(stream, map_location='cpu', weights_only=True))
            rows = [restored.step(tokens) for _ in range(2)]
            for expected, actual in zip(full_rows[1:], rows):
                for field in ('window_starts', 'lr', 'distill_weight', 'step',
                              'targets', 'schedule_digest'):
                    self.assertEqual(expected[field], actual[field])
            for name, value in full.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[name]), name)

    def test_checkpoint_average_is_fp64_accumulated_student_only(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'checkpoint-{index}.pt' for index in range(2)]
            cfg = config('D')
            models = [build_seeded_model(cfg, seed) for seed in (17, 17)]
            with torch.no_grad():
                models[1].blocks[0].qkv.weight.add_(.25)
            for index, (path, model) in enumerate(zip(paths, models)):
                torch.save(inference_checkpoint(model, cfg, 17, index + 1, 2), path)
            averaged = average_checkpoints(paths)
            expected = (models[0].blocks[0].qkv.weight.double() +
                        models[1].blocks[0].qkv.weight.double()).div(2).float()
            torch.testing.assert_close(averaged['model']['blocks.0.qkv.weight'], expected)
            self.assertNotIn('teacher', averaged)
            restored, _ = make_model(averaged['implementation'], averaged['config'], torch.device('cpu'))
            restored.load_state_dict(averaged['model'])


if __name__ == '__main__':
    unittest.main()
