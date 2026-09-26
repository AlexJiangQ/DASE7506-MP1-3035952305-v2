"""Serial GPU FP32 training / full CPU validation preflight; never scores test."""
import argparse
import gc
import json
import time
import torch
from common import ROOT, setup
from experiment_utils import WORK, write_json, append_event, check_disk, load_development_data, source_hashes
from train_experiment import Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--microbatch', type=int, default=32)
    p.add_argument('--scales', nargs='+', default=['128x4', '192x4', '192x6'])
    p.add_argument('--groups', default='ABCD')
    args = p.parse_args()
    directory = WORK / 'benchmarks' / args.tag
    if directory.exists():
        raise ValueError('Use a new benchmark tag; failed/previous attempts remain in the cost ledger.')
    check_disk()
    directory.mkdir(parents=True)
    started = time.perf_counter()
    device, _ = setup('cuda', 'fp32', 4)
    data = load_development_data()
    tokens = data['train'][0].to(device)
    results = []
    try:
        for scale in args.scales:
            for group in args.groups:
                check_disk()
                cfg = json.loads((ROOT / 'configs' / f'{group}-{scale}.json').read_text())
                engine = Engine(cfg, device=device, steps=1200, microbatch=args.microbatch)
                torch.cuda.reset_peak_memory_stats()
                durations = []
                for index in range(230):
                    row = engine.step(tokens)
                    append_event(directory / 'events.jsonl', {'event': 'benchmark_update', 'group': group, 'scale': scale, **row})
                    if index >= 30:
                        durations.append(row['seconds'])
                    if (index + 1) % 100 == 0:
                        print(f'benchmark {group}-{scale}: {index+1}/230', flush=True)
                result = engine.validate(data['validation'])
                row = {'group': group, 'scale': scale, 'microbatch': args.microbatch,
                       'warmup_updates': 30, 'measured_updates': 200,
                       'train_targets_including_warmup': 230 * 32 * 256,
                       'seconds_per_update': sum(durations) / len(durations), 'raw_seconds': durations,
                       'all_train_seconds': engine.train_seconds,
                       'validation_seconds': engine.validation_seconds,
                       'validation_score_seconds': result['seconds'],
                       'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                       'parameters': sum(p.numel() for p in engine.model.parameters())}
                results.append(row)
                write_json(directory / 'results.json', {'complete': False, 'models': results, 'source_hashes': source_hashes()})
                print(json.dumps({k: v for k, v in row.items() if k != 'raw_seconds'}), flush=True)
                del engine
                gc.collect()
                torch.cuda.empty_cache()
        write_json(directory / 'results.json', {'complete': True, 'models': results,
                   'seconds': time.perf_counter() - started, 'source_hashes': source_hashes()})
    except BaseException as error:
        append_event(directory / 'events.jsonl', {'event': 'failure', 'error': repr(error), 'seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()
