#!/usr/bin/env python3
"""Source snapshots and result checks for rerun_servers.sh (standard library only)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


COMPONENTS = {
    'repro_torch', 'repro_gemm_cuda', 'repro_sum_cuda', 'repro_math_cuda',
    'repro_conv_cuda', 'repro_nn_cuda', 'gpt2_ops_suite', 'experiments',
}
EXCLUDED = {'build', 'dist', 'results', 'results_bkp', '__pycache__', '.venv',
            'venv', '.git', 'envs', 'node_modules', 'experiment_workspace'}
EXTENSIONS = {'.py', '.cu', '.cuh', '.cpp', '.h', '.sh', '.sbatch', '.md', '.txt', '.gz'}


def wanted(path):
    return (path.parts and path.parts[0] in COMPONENTS
            and not any(p in EXCLUDED or p.endswith('.egg-info') for p in path.parts)
            and path.suffix in EXTENSIONS)


def snapshot(source, dest, metadata):
    source, dest = Path(source).resolve(), Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    if source.is_dir():
        # Walk only known source components; never traverse venvs/results/builds.
        import os
        for component in sorted(COMPONENTS):
            for base, dirs, names in os.walk(source / component, followlinks=False):
                dirs[:] = [d for d in dirs if d not in EXCLUDED
                           and not d.endswith('.egg-info') and not (Path(base) / d).is_symlink()]
                for name in names:
                    p = Path(base) / name
                    rel = p.relative_to(source)
                    if wanted(rel) and not p.is_symlink():
                        files[rel.as_posix()] = p.read_bytes()
    else:
        with tarfile.open(source, 'r:*') as archive:
            members = archive.getmembers()
            # Accept either ./repro_torch/... or a single enclosing repo folder.
            roots = {m.name[:-len('repro_torch/setup.py')] for m in members
                     if m.isfile() and m.name.endswith('repro_torch/setup.py')}
            if len(roots) != 1:
                raise ValueError('Archive must contain exactly one repro_torch/setup.py')
            prefix = roots.pop()
            for m in members:
                if not m.name.startswith(prefix):
                    continue
                rel = PurePosixPath(m.name[len(prefix):])
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe archive path: ' + m.name)
                if wanted(rel):
                    if not m.isfile():
                        if m.issym() or m.islnk():
                            raise ValueError('Source links are not accepted: ' + m.name)
                        continue
                    if str(rel) in files:
                        raise ValueError('Duplicate archive member: ' + str(rel))
                    files[str(rel)] = archive.extractfile(m).read()
    required = ['repro_torch/setup.py', 'repro_torch/mnist_repro.py',
                'repro_conv_cuda/repro_conv.cu', 'repro_gemm_cuda/repro_gemm.cu']
    for name in required:
        if name not in files:
            raise ValueError('Missing required source: ' + name)
    manifest = {}
    atomic = []
    for name, data in sorted(files.items()):
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        p.chmod(0o755 if p.suffix in ('.sh', '.sbatch') else 0o644)
        manifest[name] = hashlib.sha256(data).hexdigest()
        if p.suffix in ('.cu', '.cuh'):
            for line_no, line in enumerate(data.decode().splitlines(), 1):
                if re.search(r'\batomic\w*\s*\(', line):
                    atomic.append(f'{name}:{line_no}')
    record = {'source': str(source), 'files': manifest, 'explicit_atomic_call_lines': atomic}
    if source.is_file():
        record['source_archive_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    Path(metadata).write_text(json.dumps(record, indent=2) + '\n')
    print(f'Snapshotted {len(files)} files; {len(atomic)} explicit CUDA atomic call lines')


def collect(source, output):
    for component in COMPONENTS:
        p = Path(source) / component / 'results'
        if p.is_dir():
            shutil.copytree(p, Path(output) / component, dirs_exist_ok=True)


def check_runs(root, expected_file):
    """Check completion and repeated hashes, without comparing different seeds/batches."""
    expected = Path(expected_file).read_text().splitlines()
    problems, groups = [], {}
    for line in expected:
        path, mode, seed, batch, profile = line.split('\t')
        p = Path(path).with_suffix('.json')
        try:
            d = json.loads(p.read_text())
            wanted_values = (mode, int(seed), int(batch), bool(int(profile)), 2, 3)
            values = tuple(d[k] for k in ('mode', 'seed', 'batch', 'profile', 'epochs', 'slices'))
            if values != wanted_values or d['steps'] != 2 * (60000 // int(batch)):
                raise ValueError('wrong configuration or incomplete training')
            if 'weights_epoch_2' not in d['hashes'] or 'hash data ' not in Path(path).read_text():
                raise ValueError('missing completion hashes')
            if mode == 'repro':
                key = (int(seed), int(batch), d['slices'])
                groups.setdefault(key, set()).add(json.dumps(d['hashes'], sort_keys=True))
        except (OSError, ValueError, KeyError) as error:
            problems.append(f'{p}: {error}')
    for key, hashes in groups.items():
        if len(hashes) != 1:
            problems.append(f'Repro hashes differ for seed,batch,slices={key}')
    summary = {'expected_training_runs': len(expected), 'problems': problems,
               'repro_hash_groups_checked': len(groups), 'passed': not problems}
    Path(root, 'run_checks.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return bool(problems)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['snapshot', 'collect', 'check'])
    ap.add_argument('paths', nargs='+')
    args = ap.parse_args()
    if args.action == 'snapshot':
        snapshot(*args.paths)
    elif args.action == 'collect':
        collect(*args.paths)
    else:
        raise SystemExit(check_runs(*args.paths))


if __name__ == '__main__':
    main()
