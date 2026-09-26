"""Fresh-process CPU FP32 validation timing, peak RAM and inference asset accounting."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import torch
from common import PROTOCOL, make_model, setup, sha
from evaluate import score
from experiment_utils import load_development_data, peak_ram_bytes, write_json


def worker(checkpoint_path, output):
    device, _ = setup('cpu', 'fp32', 4)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('protocol') != PROTOCOL:
        raise ValueError('Checkpoint protocol mismatch.')
    model, implementation_sha = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    data = load_development_data()
    result = score(model, *data['validation'], device, 'fp32')
    result.pop('window_nll_nats')
    result.update(checkpoint_sha256=sha(checkpoint_path), implementation_sha256=implementation_sha,
                  peak_process_ram_bytes=peak_ram_bytes())
    write_json(output, result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--asset', action='append', type=Path, default=[])
    p.add_argument('--worker-checkpoint', type=Path)
    p.add_argument('--worker-output', type=Path)
    args = p.parse_args()
    if args.worker_checkpoint:
        worker(args.worker_checkpoint, args.worker_output)
        return
    if not args.checkpoint or not args.baseline or not args.output_dir or args.repeats < 1:
        p.error('Parent mode requires checkpoint, baseline, output-dir and positive repeats.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error('Output directory must be new or empty.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for label, checkpoint in (('baseline', args.baseline), ('candidate', args.checkpoint)):
        rows = []
        for index in range(1, args.repeats + 1):
            output = args.output_dir / f'{label}-rep{index}.json'
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--worker-checkpoint', str(checkpoint.resolve()),
                            '--worker-output', str(output.resolve())], check=True)
            rows.append(json.loads(output.read_text()))
        records[label] = rows
    baseline_seconds = [row['seconds'] for row in records['baseline']]
    candidate_seconds = [row['seconds'] for row in records['candidate']]
    assets = [args.checkpoint, *args.asset]
    asset_rows = [{'path': str(path.resolve()), 'bytes': path.stat().st_size,
                   'sha256': sha(path)} for path in assets]
    summary = {'protocol': 'validation-only-resource-qualification',
               'repeats': args.repeats, 'threads': 4, 'precision': 'fp32',
               'baseline_seconds': baseline_seconds, 'candidate_seconds': candidate_seconds,
               'baseline_median_seconds': statistics.median(baseline_seconds),
               'candidate_median_seconds': statistics.median(candidate_seconds),
               'cpu_ratio': statistics.median(candidate_seconds) / statistics.median(baseline_seconds),
               'candidate_peak_ram_bytes': max(row['peak_process_ram_bytes'] for row in records['candidate']),
               'assets': asset_rows, 'asset_bytes': sum(row['bytes'] for row in asset_rows),
               'test_run': False}
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
