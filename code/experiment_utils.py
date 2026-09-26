"""Local experiment I/O, provenance, independent RNGs and development data."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT, PROTOCOL, sha

WORK = ROOT.parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def append_event(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps({'time': time.time(), **value}, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def check_disk():
    free = shutil.disk_usage(WORK).free
    if free < 5 * 1024**3:
        raise RuntimeError(f'Less than 5 GiB free: {free / 1024**3:.2f} GiB. No files were deleted.')
    return free


def source_hashes():
    files = ['student.py', 'train_experiment.py', 'distill_experiment.py',
             'checkpoint_average.py', 'measure_resources.py', 'experiment_utils.py',
             'model.py', 'common.py', 'evaluate.py']
    return {f: sha(ROOT / f) for f in files}


def load_development_data():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        if sha(ROOT / 'data' / name) != expected:
            raise ValueError(f'Changed benchmark file: {name}')
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
    result = {}
    # Hash verification covers the package, but development never tokenizes/scorers test.
    for split in ('train', 'validation'):
        raw = (ROOT / 'data' / f'wikitext_{split}.txt').read_bytes()
        result[split] = (torch.tensor(tokenizer.encode(raw.decode('utf-8')).ids, dtype=torch.long), len(raw))
    return result


def rng_state(data_generator):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'data_sampler': data_generator.get_state()}


def restore_rng(state, data_generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    data_generator.set_state(state['data_sampler'])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def inference_checkpoint(model, config, seed, updates, batch_size, implementation='student'):
    return {'protocol': PROTOCOL, 'implementation': implementation, 'config': dict(config),
            'model': cpu_tree(model.state_dict()), 'seed': seed, 'updates': updates,
            'train_tokens': updates * batch_size * 256}


def peak_ram_bytes():
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource
    import sys
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)
