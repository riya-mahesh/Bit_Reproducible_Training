"""Test orchestration with simulated hardware; never launch CUDA locally."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class A100RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        for name in ('repro_torch/setup.py', 'repro_torch/mnist_repro.py',
                     'repro_torch/repro_torch/__init__.py',
                     'repro_conv_cuda/repro_conv.cu', 'repro_gemm_cuda/repro_gemm.cu'):
            p = self.repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('// source\n' if p.suffix == '.cu' else '# source\n')
        scripts = self.repo / 'experiments'
        scripts.mkdir()
        for name in ('run_a100_no_atomics.sh', 'rerun_support.py'):
            shutil.copy2(HERE / name, scripts / name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name, body in dict(hostname='echo a100-test', nvcc='echo mock-nvcc',
                               lscpu='echo mock-cpu', **{'nvidia-smi': 'echo mock-A100'}).items():
            self.write_exe(self.bin / name, '#!/bin/bash\n' + body + '\n')
        # Let all source, hash, completeness and archive checks execute normally.
        self.py = self.bin / 'fake_python'
        self.write_exe(self.py, '#!' + sys.executable + '\n' + r'''
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == '-' and len(a) == 1:
    code = sys.stdin.read()
    assert 'torch.cuda' in code
    print('mock A100 preflight'); sys.exit(0)
if a[0] == '-' and a[1].endswith('loaded_extension.json'):
    sys.stdin.read()
    p = Path(os.environ['REPRO_BUILD']) / 'repro_torch/mock.so'
    Path(a[1]).write_text(json.dumps({'path': str(p)})); sys.exit(0)
if a[:3] == ['-m', 'pip', 'freeze']:
    print('torch==2.6.0+cu124'); sys.exit(0)
if a[0] == 'setup.py':
    p = Path(a[a.index('--build-lib') + 1]) / 'repro_torch/mock.so'
    p.write_text('fresh extension'); sys.exit(0)
if Path(a[0]).name == 'mnist_repro.py':
    def get(k): return a[a.index(k) + 1]
    mode, seed, batch = get('--mode'), int(get('--seed')), int(get('--batch'))
    if os.environ.get('FAIL_MODE') == mode: sys.exit(9)
    smoke = '--max-steps' in a
    h = dict(weights_epoch_1='7fed5293c5306f9c9ec9879b7a4c5ad35cfacedbc5090a6b18801e62a3360d9d',
             weights_epoch_2='6b8ce5843c58dcd10a95e3339e634b66f95989e2972cc475427c8fb7b83ab2f2',
             test_logits='5f22405beb10931b28774aa7b2e560085ca15598d91d96e5f79b8781da1b357a',
             train_logits='ada4cbfcdb1f4cdb9cdf9de2d1c82389a7153e18feb3f87348755325be957a47')
    if smoke:
        h['weights_epoch_1'] = '286b280c7e33bcb70aa2bf329f1e01a8e7764765e90f2305f478afdbe895f64d'
        if os.environ.get('BAD_SMOKE'): h['weights_epoch_1'] = 'wrong'
    d = dict(mode=mode, seed=seed, batch=batch, epochs=2, slices=3, hashes=h,
             profile='--profile' in a, steps=10 if smoke else 2*(60000//batch))
    p = Path(get('--out')); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('hash data fixed\n')
    p.with_suffix('.json').write_text(json.dumps(d)); sys.exit(0)
os.execv(sys.executable, [sys.executable] + a)
''')
        self.write_exe(self.bin / 'ncu', '#!' + sys.executable + '\n' + r'''
import sys
from pathlib import Path
a = sys.argv[1:]
if '--export' in a:
    Path(a[a.index('--export') + 1] + '.ncu-rep').write_text('mock report')
if '--import' in a:
    assert Path(a[a.index('--import') + 1]).is_file()
print('mock ncu')
''')
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        RUN_ID='test', A100_RESULTS_ROOT=str(self.root / 'results'),
                        A100_BUILD_ROOT=str(self.root / 'build'))

    @staticmethod
    def write_exe(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_script(self, *args, **env):
        return subprocess.run(['bash', str(self.repo / 'experiments/run_a100_no_atomics.sh'),
                               '--python', str(self.py), *args],
                              cwd=self.repo / 'repro_torch', env=dict(self.env, **env),
                              capture_output=True, text=True)

    def test_full_protocol_archive_and_isolation(self):
        old = self.repo / 'repro_torch/repro_torch/old.so'
        old.write_text('old extension')
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = self.root / 'results/a100-test/test'
        checks = json.loads((out / 'run_checks.json').read_text())
        self.assertTrue(checks['passed'])
        self.assertEqual(checks['expected_training_runs'], 43)
        self.assertEqual(len(list((out / 'profiling').glob('*.ncu-rep'))), 4)
        self.assertEqual((out / 'status.txt').read_text().strip(), 'COMPLETE')
        self.assertEqual(old.read_text(), 'old extension')
        archive = out.with_suffix('.tar.gz')
        expected = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertTrue(Path(str(archive) + '.sha256').read_text().startswith(expected))
        with tarfile.open(archive) as t:
            self.assertEqual(t.extractfile('test/status.txt').read(), b'COMPLETE\n')
            self.assertIn('test/source.tar.gz', t.getnames())
        self.assertNotEqual(self.run_script().returncode, 0)

    def test_failures_are_not_reported_as_success(self):
        for name, env in [('missing_repdl_runs', dict(FAIL_MODE='repdl')),
                          ('bad_smoke', dict(BAD_SMOKE='1'))]:
            r = self.run_script('--skip-ncu', RUN_ID=name, **env)
            self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
            out = self.root / 'results/a100-test' / name
            self.assertEqual((out / 'status.txt').read_text().strip(), 'FAILED')
            self.assertTrue(out.with_suffix('.tar.gz').is_file())
        (self.repo / 'repro_conv_cuda/repro_conv.cu').write_text('atomicMax(&m, x);\n')
        r = self.run_script('--skip-ncu', RUN_ID='atomic')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Refusing old GPU atomic source',
                      (self.root / 'results/a100-test/atomic/runner.log').read_text())
        self.assertFalse((self.root / 'build/a100-test/atomic/package').exists())


if __name__ == '__main__':
    unittest.main()
