"""Write provenance/configuration artifacts without altering supplied source files."""
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import torch
import numpy as np
import tokenizers
from common import ROOT, sha
from experiment_utils import WORK, write_json, check_disk


def main():
    check_disk()
    provenance = WORK / 'provenance'
    provenance.mkdir(exist_ok=True)
    original = WORK / 'original'
    hashes = {str(p.relative_to(original)).replace('\\', '/'): sha(p)
              for p in original.rglob('*') if p.is_file()}
    original_manifest = provenance / 'original_sha256.json'
    if original_manifest.exists():
        if json.loads(original_manifest.read_text()) != hashes:
            raise ValueError('Original snapshot changed.')
    else:
        write_json(original_manifest, hashes)
    for name in ('model.py', 'train.py', 'common.py', 'evaluate.py', 'requirements.txt', 'tests/test_contract.py'):
        assert sha(ROOT / name) == sha(original / 'code' / name), name
    for path in (ROOT / 'data').rglob('*'):
        if path.is_file():
            assert sha(path) == sha(original / 'code' / path.relative_to(ROOT)), str(path)
    for width, depth in ((128, 4), (192, 4), (192, 6)):
        for group in 'ABCD':
            cfg = dict(vocab=2048, context=256, width=width, depth=depth, heads=4,
                       position_encoding='rope' if group in 'BD' else 'learned',
                       ffn_type='swiglu' if group in 'CD' else 'gelu')
            path = ROOT / 'configs' / f'{group}-{width}x{depth}.json'
            if path.exists():
                assert json.loads(path.read_text()) == cfg, str(path)
            else:
                write_json(path, cfg)
    environment = {'captured_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'python': sys.version,
                   'executable': sys.executable, 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'numpy': np.__version__, 'tokenizers': tokenizers.__version__,
                   'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
                   'free_disk_bytes': check_disk(),
                   'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    if torch.cuda.is_available():
        environment.update(gpu=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        x = torch.randn(64, 64, device='cuda')
        assert torch.isfinite(x @ x).all()
        environment['cuda_fp32_matmul_check'] = 'passed'
    path = provenance / f'environment-{time.strftime("%Y%m%d-%H%M%S")}.json'
    write_json(path, environment)
    print(json.dumps(environment, indent=2))
    print(f'Original snapshot: {len(hashes)} files verified; protected working files unchanged.')


if __name__ == '__main__':
    main()
