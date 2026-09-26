"""Persist unittest output and runtime, including failing development checks."""
import argparse
import io
import sys
import time
import unittest
from experiment_utils import WORK, write_json, source_hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    output = WORK / 'checks' / args.tag
    if output.exists():
        raise ValueError('Use a fresh check tag; prior checks are retained.')
    output.mkdir(parents=True)
    start = time.perf_counter()
    with (output / 'unittest.txt').open('w', encoding='utf-8') as log:
        suite = unittest.defaultTestLoader.discover('tests')
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'skipped': len(result.skipped), 'passed': result.wasSuccessful(),
               'seconds': time.perf_counter() - start, 'source_hashes': source_hashes()}
    write_json(output / 'results.json', summary)
    print(summary, flush=True)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
