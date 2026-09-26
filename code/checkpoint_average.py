"""Average compatible inference checkpoints into one student-only FP32 checkpoint."""
import argparse
from pathlib import Path
import torch
from common import PROTOCOL, sha
from experiment_utils import check_disk, write_json


def average_checkpoints(paths):
    paths = [Path(path) for path in paths]
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
    first = checkpoints[0]
    for checkpoint in checkpoints:
        for key in ('protocol', 'implementation', 'config', 'seed'):
            if checkpoint.get(key) != first.get(key):
                raise ValueError(f'Incompatible checkpoint field: {key}')
        if checkpoint['protocol'] != PROTOCOL:
            raise ValueError('Checkpoint protocol mismatch.')
        if checkpoint['model'].keys() != first['model'].keys():
            raise ValueError('Checkpoint parameter names differ.')
    averaged = {}
    for name in first['model']:
        tensors = [checkpoint['model'][name] for checkpoint in checkpoints]
        if any(tensor.shape != tensors[0].shape or tensor.dtype != tensors[0].dtype for tensor in tensors):
            raise ValueError(f'Incompatible tensor: {name}')
        if tensors[0].is_floating_point():
            averaged[name] = torch.stack([tensor.double() for tensor in tensors]).mean(0).float()
        else:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f'Non-floating tensor differs: {name}')
            averaged[name] = tensors[0].clone()
    result = {key: first[key] for key in ('protocol', 'implementation', 'config', 'seed')}
    result.update(model=averaged,
                  updates=max(checkpoint.get('updates', 0) for checkpoint in checkpoints),
                  train_tokens=max(checkpoint.get('train_tokens', 0) for checkpoint in checkpoints),
                  averaged_checkpoints=[{'path': str(path.resolve()), 'sha256': sha(path),
                                         'updates': checkpoint.get('updates')}
                                        for path, checkpoint in zip(paths, checkpoints)])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if len(args.checkpoint) < 2:
        p.error('At least two checkpoints are required.')
    if args.output.exists():
        p.error('Refusing to overwrite an existing averaged checkpoint.')
    check_disk()
    result = average_checkpoints(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    write_json(args.output.with_suffix('.json'), {
        'checkpoint_sha256': sha(args.output),
        'inputs': result['averaged_checkpoints'],
        'implementation': result['implementation'], 'config': result['config']})


if __name__ == '__main__':
    main()
