# DASE7506 MP1 - Distilled Rotary Language Model

This repository contains the code, report, fixed data pipeline, configurations,
and evidence for the submitted MP1 predictor. The matching checkpoint is
distributed separately as `MP1_checkpoint_bundle_20260926_FINAL.zip` so it can be
downloaded and evaluated without retraining.

## Final result

- Full-test BPB: **1.5194970322432222** (lower is better).
- Original A/seed-17/step-1200 baseline test BPB: **2.101258060349007**.
- Frozen student: width 288, depth 6, four heads, RoPE, SwiGLU hidden width 768.
- Parameters: 6,587,136.
- CPU FP32 test-time ratio: **4.154717x** (limit: less than 5x).
- Peak test process-tree RAM: **1,868,443,648 bytes / 1.74 GiB** (limit: 4 GiB).
- Uncompressed inference assets: **28,874,091 bytes / 27.54 MiB** (limit: 64 MiB).
- Frozen checkpoint SHA-256:
  `5414d777646765f4429b7f59af832edb92df2acb5e17d32cd33a41d67d191b4e`.

The method, averaging rule, and assets were frozen before test evaluation. Test
results were never used to choose architecture, seed, update count, or weights.

## Contents

- `code/`: official-compatible evaluator plus model, training, distillation,
  checkpoint averaging, resource measurement, configurations, tests, and the
  supplied fixed dataset/tokenizer.
- `report/MP1_Report.pdf`: final six-page report (under the 10-page limit).
- `evidence/`: test JSON, resource record, three-seed ablation summary, capacity
  summary, matched long-run summary, protocol, and complete cost ledger.
- `guide/GUIDE.md`: supplied assignment guide.

## Environment

The audited environment used Python 3.12.3, PyTorch 2.7.1+cu126, NumPy 2.5.3,
and tokenizers 0.21.4. CPU evaluation does not require a CUDA toolkit.

PowerShell:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r '.\code\requirements.txt'
```

Linux/macOS shell:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
```

For GPU training on the audited CUDA 12.6 setup, install the official PyTorch
CUDA wheel before the remaining requirements:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
& '.\.venv\Scripts\python.exe' -m pip install numpy==2.5.3 tokenizers==0.21.4
```

## Reproduce the submitted score without retraining

Extract the code and checkpoint archives beside each other:

```text
parent/
  code_repository/
  checkpoint_bundle/
```

From `code_repository`, run:

```powershell
& '.\.venv\Scripts\python.exe' '.\code\evaluate.py' `
  --checkpoint '..\checkpoint_bundle\checkpoint.pt' `
  --device cpu --precision fp32 --threads 4 --split test `
  --output '.\reproduced_test.json'
```

Expected BPB: `1.5194970322432222`. Small timing differences are normal; BPB
must reproduce under the fixed FP32 scorer. The evaluator also writes per-window
NLL values beside the JSON output.

Before evaluation, verify the checkpoint:

```powershell
(Get-FileHash -Algorithm SHA256 '..\checkpoint_bundle\checkpoint.pt').Hash.ToLower()
```

## Reproduce the final training lineage

All commands below use the supplied train split for gradients/teacher queries
and validation only for predeclared selection. Every model starts from random
initialization. Run from the repository root after installing CUDA PyTorch.

Train the two D-448x6 teachers:

```powershell
& '.\.venv\Scripts\python.exe' '.\code\train_experiment.py' `
  --config '.\code\configs\D-448x6.json' `
  --run-dir '.\reproduction\teacher-s17' --seed 17 --steps 6000 `
  --batch-size 32 --microbatch 32 --lr 0.001 --device cuda `
  --threads 4 --eval-every 600

& '.\.venv\Scripts\python.exe' '.\code\train_experiment.py' `
  --config '.\code\configs\D-448x6.json' `
  --run-dir '.\reproduction\teacher-s42' --seed 42 --steps 6000 `
  --batch-size 32 --microbatch 32 --lr 0.001 --device cuda `
  --threads 4 --eval-every 600
```

Each teacher selects the lowest validation BPB on the predeclared
`1200, 1800, ..., 6000` grid (ties choose the earlier checkpoint). Train the
student with a 50:50 arithmetic mean of teacher probabilities:

```powershell
& '.\.venv\Scripts\python.exe' '.\code\distill_experiment.py' `
  --config '.\code\configs\D-288x6.json' `
  --teacher '.\reproduction\teacher-s17\selected.pt' `
  --teacher '.\reproduction\teacher-s42\selected.pt' `
  --distill-weight 0.50 --temperature 2 `
  --run-dir '.\reproduction\student-s17' --seed 17 --steps 6000 `
  --batch-size 32 --microbatch 32 --lr 0.001 --device cuda `
  --threads 4 --eval-every 600 --save-at 4200 4800 5400 6000
```

Average the four fixed student checkpoints in FP64 and export FP32:

```powershell
& '.\.venv\Scripts\python.exe' '.\code\checkpoint_average.py' `
  --checkpoint '.\reproduction\student-s17\checkpoint-step4200.pt' `
  --checkpoint '.\reproduction\student-s17\checkpoint-step4800.pt' `
  --checkpoint '.\reproduction\student-s17\checkpoint-step5400.pt' `
  --checkpoint '.\reproduction\student-s17\checkpoint-step6000.pt' `
  --output '.\reproduction\student-averaged.pt'
```

GPU kernels can produce very small platform-specific training differences. The
submitted predictor is the frozen checkpoint bundle; retraining is not required
for score verification.

## Tests

From `code/`:

```powershell
& '..\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

The final suite contained 23 passing tests covering causality, normalized
probabilities, all four architecture variants, shared initialization, finite
gradients, independent sampler RNG, gradient accumulation, checkpoint loading,
and exact CPU resume behavior.

## Data and method attribution

WikiText-2 is attributed to Merity et al., *Pointer Sentinel Mixture Models*.
The supplied data notices identify CC BY-SA 3.0 and the GNU Free Documentation
License. RoPE follows Su et al. (2021), SwiGLU follows Shazeer (2020), and
knowledge distillation follows Hinton, Vinyals, and Dean (2015). No external
training text or pretrained weights were used.

## AI assistance disclosure

AI assistance was used to discuss the feasibility of candidate methods, correct
conceptual misunderstandings, and support implementation and experimental
verification. All model weights were trained locally from random initialization
using only the supplied training split.
