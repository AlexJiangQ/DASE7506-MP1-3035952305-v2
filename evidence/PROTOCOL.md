# Preregistered experiment protocol

Status: fixed before formal training on 2026-09-20. Protocol: 7506-mp1-wt2-v2.

## Primary factorial study

| Group | Position | FFN |
|---|---|---|
| A | Learned absolute | GELU, hidden 4d |
| B | RoPE, base 10000 | GELU, hidden 4d |
| C | Learned absolute | SwiGLU, hidden nearest multiple of 8 to 8d/3 |
| D | RoPE, base 10000 | SwiGLU, same width rule |

Width 128, depth 4, heads 4, context 256, vocab 2048; tied input/output weights,
LayerNorm, no dropout. Seeds 17, 42, 123. Each run starts from random initialization
and processes 1200 * 32 * 256 = 9,830,400 training targets. GPU FP32, official
AdamW (weight decay .1), 100-update warmup, cosine schedule, peak LR .001,
gradient norm cap 1. Full CPU FP32 validation every 300 updates, four threads.
Endpoint comparison always uses update 1200, never the best intermediate point.

Canonical untrained baseline initializes all semantically matching tensors.
Model seed = data seed = seed; training/dropout RNG seed = seed + 100000.
Unique tensors use stable SHA256-derived private seeds keyed by parameter name
and seed + 10000. Data uses its own CPU generator. Paired runs use the same
microbatch; if needed, use 16 or 8 with accumulation to preserve effective 32.

Report sample mean, sample SD (ddof=1), paired B-A, C-A, D-A and D-C-B+A.
Negative BPB effects mean improvement. Three seeds do not establish significance
or independence. Choose B/C/D by mean validation BPB; candidates within 0.002 of
the minimum are ordered by median CPU validation time, then parameter count.
This is an engineering selection rule, not a significance test.

## Conditional extension

Benchmark every planned architecture/scale independently: 30 warmup + 200 measured
GPU updates and a full CPU validation. All benchmark targets/time count as cost.
Formal runs restart initialization. Predict model-specific update time, CPU
validation, saving/loading and preparation; do not extrapolate only from 128x4.

- <=20 h full prediction: scales 128x4,192x4,192x6; LR .001/.0006;
  screens 3600, seed17; up to two candidates retrained from scratch to 6000;
  winner additional seeds42/123 after matched controls.
- 20–30 h: omit192x6; two LRs; only first candidate enters6000.
- 30–40 h: scales128x4/192x4, LR .001,3600,seed17 candidate/control only.
- >40 h: core study; at most128x4 candidate/control3600 if time permits.

Reestimate after narrowing and step down until within24 planned hours, with6
hours reserved for recovery/reproduction. Also obey3–6h/day and Sept27 freeze.
Long controls use identical scale, LR, target budget, data schedule and selection
rule, with learned/GELU. The original128x4 baseline remains the resource reference.

For T=3600 or6000, select submission seed17 checkpoint only from updates
{1200,1800,2400,...,T}, minimum full CPU validation BPB, exact ties earlier.
Report this separately from the fixed-T endpoint, including both selected-point
targets and whole-run/search costs. If no extension, candidates are core endpoints.

## Resources, test and deliverables

Validation timing: official scorer, CPU FP32,4 threads, training stopped;
each model once warmed and five scored repetitions; store all times and median.
Separately record process peak RAM from load to score and uncompressed inference
asset bytes. Internal limits4x/3GiB/48MiB; official limits5x/4GiB/64MiB.
Validation limits do not substitute for frozen full-test workload confirmation.

Before first new test: freeze config, weights, code/assets and hashes, then
reproduce validation in a clean environment. Score only frozen final predictor
and original baseline on full CPU FP32 test, once by default. Reproduction reruns
require a recorded reason and cannot inform further selection. Failures are not
silently removed. Do not run the original README's test-split smoke command.

Sept27 freeze/test; Sept28 report/materials/first submission; Sept29 link fixes;
Sept30 final check. English report <=10 pages including references; disclose
method sources, all training/search costs, failures, limitations and AI assistance.
The user handles code publication, immutable URL and course submission/Issue.
Public peer review is a later user-directed action.
