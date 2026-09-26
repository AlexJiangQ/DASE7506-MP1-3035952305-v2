# MP1 experiment cost ledger

Updated: 2026-09-26. Test split scorer runs: **5** (baseline once, the identical
frozen candidate four times). Model selection was locked before every test run.

## Campaign totals

- Formal run directories with `metrics.json`: 38 (excluding the CPU smoke run).
- Formal optimizer updates: 105,600.
- Recorded GPU training time: 5.241 hours.
- Recorded full CPU validation time: 1.824 hours.
- These totals exclude environment setup, unit tests, short GPU diagnostics,
  checkpoint I/O, reporting time, and fresh-process resource qualification;
  those artifacts are retained separately rather than silently omitted.
- No overnight formal run failed or required an OOM retry.

The full per-run machine-readable records remain in each `runs/*/metrics.json`
and `events.jsonl`. Earlier diagnostic and screening costs are documented in
`HANDOFF_ZH.md` and the benchmark/smoke directories.

## Overnight decision runs

| Run | Updates | GPU train (s) | CPU validation (s) | Selected step | Selected BPB |
|---|---:|---:|---:|---:|---:|
| A-288x6 matched | 3,600 | 533.4 | 242.3 | 3,600 | 1.610690 |
| D-288x6 matched | 3,600 | 605.2 | 268.3 | 3,000 | 1.638177 |
| D-448x6 screen | 1,200 | 370.3 | 162.3 | 1,200 | 1.667470 |
| D-448x6 teacher seed17 | 6,000 | 1,854.4 | 831.6 | 2,400 | 1.618143 |
| D-448x6 teacher seed42 | 6,000 | 1,850.1 | 837.2 | 2,400 | 1.624628 |
| Single teacher, lambda=0.30 | 3,600 | 988.6 | 251.2 | 3,600 | 1.542787 |
| Single teacher, lambda=0.50 | 3,600 | 988.6 | 251.7 | 3,600 | 1.525943 |
| Double teacher, lambda=0.50 | 3,600 | 1,355.9 | 254.6 | 3,600 | 1.519575 |
| Final ordinary D-288x6 | 6,000 | 996.0 | 423.3 | 3,000 | 1.654126 |
| Final double-distilled D-288x6 | 6,000 | 2,258.4 | 430.3 | 6,000 | 1.500414 |
| Double-teacher A-288x6 extension | 3,600 | 1,345.8 | 293.5 | 3,600 | 1.530341 |
| D-288x6 lambda 0.50-to-0.20 pilot | 3,600 | 1,407.9 | 297.9 | 3,600 | 1.543299 |

The fixed average of student steps {4200, 4800, 5400, 6000} achieved validation
BPB 1.494533. Averaging and resource measurements add CPU/I/O cost but no
optimizer updates.

Two cheaper averaging subsets were evaluated after freezing and retained on D: the
{5400, 6000} average scored 1.497331 and the {4800, 5400, 6000} average scored
1.495849. Neither displaced the preregistered four-checkpoint average.

## Resource qualification cost

The complete frozen average and original A-128x4 baseline were each warmed once
and then scored five times in fresh CPU FP32 processes on validation. Candidate
timed runs totalled 206.20 seconds and baseline timed runs totalled 46.42 seconds,
excluding warmups/process setup. Raw final values are in
`resources/final-D288-averaged-validation-v2/summary.json`.

## Formal test evaluation

The first paired CPU FP32 test run scored the original A/seed17/step1200 baseline
at 2.101258 BPB in 12.5708 seconds and the frozen candidate at 1.519497 BPB in
52.2281 seconds, a 4.154717x ratio. The candidate's calibrated process-tree peak
working set was 1,868,443,648 bytes. Inference assets remained 28,874,091 bytes.

The candidate test was repeated twice solely because the first two PowerShell RAM
monitors observed the small virtual-environment launcher instead of its child
Python process. A validation-only calibration established the process-tree method;
the final candidate repeat was separately authorized by the user. All three
candidate scores were exactly 1.5194970322432222. No test result changed model
selection. Raw records are under `results/formal-test-20260924/`.

On 26 September, the user explicitly requested one final CPU-performance-mode BPB
reproduction. The identical frozen candidate again scored exactly
1.5194970322432222; its isolated timing was slower and was not used to replace the
matched test-time ratio. This brought the transparent total to five scorer runs.

## Known failed or non-selected work

- Initial development had one unit-test failure in SwiGLU private-parameter
  initialization; it was fixed and retained in `IMPLEMENTATION_REVIEW.md`.
- The first baseline launcher returned `-1` before creating a run directory;
  the successful rerun and absence of training cost are documented in
  `HANDOFF_ZH.md`.
- Many valid models were not selected: A/D matched controls, capacity screens,
  both single-teacher pilots, and the unaveraged final checkpoint. Their costs
  are included above/in campaign totals and their files were not deleted.
- D-448x6 teachers overfit after step 2400. Full 6000-step costs are reported;
  distillation used validation-selected step-2400 checkpoints.
- The first frozen-directory reproduction check failed before scoring because
  `model.py`, a runtime dependency of `student.py`, was omitted from the first
  bundle copy. No split was read. The file was added, a finite/normalized fixed
  input forward check passed, and the complete asset set was re-qualified in a
  new v2 resource directory. The incomplete v1 measurement was retained.
- The final unittest command was first invoked from `MP1_work` instead of
  `MP1_work/code`, so test imports failed immediately (`common` was not on the
  module path). The command was rerun from the documented working directory;
  all 22 tests passed. No model training or split scoring occurred in the failed
  invocation.
- A double-teacher A-288x6 student scored 1.530341 at 3600 steps, 0.010767 worse
  than the matched D-288x6 double-teacher pilot. It was not extended to 6000 steps.
- A D-288x6 double-teacher pilot with linear distillation-weight annealing from
  0.50 to 0.20 scored 1.543299 at 3600 steps, 0.023724 worse than the constant
  0.50 pilot. It was not extended. The post-change full suite passed 23 tests.
- The annealing pilot launcher failed twice before training: once due to obsolete
  CLI argument names, and once due to an incorrect config path. Both failures were
  retained; the successful fresh run used a new `-v2` directory.
