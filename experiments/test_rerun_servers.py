"""Local orchestration checks using fake GPU/build commands, not numerical GPU tests.

Run: python3 -m unittest discover -s experiments -p test_rerun_servers.py -v
"""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rerun_support', HERE / 'rerun_support.py')
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        for name in ['repro_torch/setup.py', 'repro_torch/mnist_repro.py',
                     'repro_torch/repro_torch/__init__.py', 'repro_conv_cuda/repro_conv.cu',
                     'repro_gemm_cuda/repro_gemm.cu']:
            p = self.source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('// atomicAdd(&a, 1);\n' if p.suffix == '.cu' else '# fixture\n')
        (self.source / 'repro_torch/repro_torch/old_backend.so').write_text('untouched')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        commands = {
            'hostname': 'printf "%s\\n" "${FAKE_HOST:-cuda2}"',
            'nvcc': 'echo "Cuda compilation tools, release 12.4, V12.4.131"',
            'nvidia-smi': 'echo "0"',
            'lscpu': 'echo "mock CPU"',
        }
        for name, body in commands.items():
            self.executable(self.bin / name, '#!/bin/bash\n' + body + '\n')
        venv = self.root / 'envs/cuda124'
        (venv / 'bin').mkdir(parents=True)
        (venv / 'config').write_text("CUDA_MODULE=''\n")
        (venv / 'bin/activate').write_text('# no environment changes in test\n')
        official = venv / 'RepDL/examples/mnist_trainig.py'
        official.parent.mkdir(parents=True)
        official.write_text('print("mock official example")\n')
        # Intercept only hardware/training operations. Run helper scripts normally.
        fake_python = '''import sys, os, json
from pathlib import Path
a = sys.argv[1:]
if a[0] == '-':
    sys.stdin.read(); sys.exit(0)
if a[0] in ('-c', '-m'):
    print('mock dependency/device metadata'); sys.exit(0)
if a[0] == 'setup.py':
    p = Path(a[a.index('--build-lib') + 1]) / 'repro_torch/mock.so'
    p.write_text('mock compiled output'); sys.exit(0)
if Path(a[0]).name == 'mnist_repro.py':
    options = dict(zip(a[1::2], a[2::2]))
    mode, seed, batch = options['--mode'], int(options['--seed']), int(options['--batch'])
    if os.environ.get('FAIL_MODE') == mode:
        sys.exit(9)
    p = Path(options['--out']); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('hash data fixed\\n')
    d = dict(mode=mode, seed=seed, batch=batch, epochs=2, slices=3,
             profile='--profile' in a, steps=2*(60000//batch),
             hashes={'weights_epoch_2': f'{seed}-{batch}'})
    p.with_suffix('.json').write_text(json.dumps(d)); sys.exit(0)
os.execv(sys.executable, [sys.executable] + a)
'''
        self.executable(venv / 'bin/python3', '#!' + sys.executable + '\n' + fake_python)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        PYTHON=sys.executable, REPRO_WORK=str(self.root / 'work'),
                        REPRO_ENV_ROOT=str(self.root / 'envs'),
                        RESULTS_ROOT=str(self.root / 'results'), RUN_ID='test')
        self.env.pop('CUDA_VISIBLE_DEVICES', None)

    @staticmethod
    def executable(path, data):
        path.write_text(data)
        path.chmod(0o755)

    def run_script(self, *args, **env):
        return subprocess.run(['bash', str(HERE / 'rerun_servers.sh'), '--source', str(self.source), *args],
                              env=dict(self.env, **env), capture_output=True, text=True)

    def test_host_matrix(self):
        for host, count in [('cuda2', 3), ('cuda3', 2), ('cuda5', 3)]:
            r = self.run_script('--dry-run', FAKE_HOST=host)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(r.stdout.count('Config cuda'), count)
        self.assertNotEqual(self.run_script('--dry-run', FAKE_HOST='cuda4').returncode, 0)

    def test_cims_isolation_and_completion(self):
        for host in ('cuda2', 'cuda3'):
            r = self.run_script('--configs', 'cuda124', '--gpus', '0 1', FAKE_HOST=host)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            out = self.root / 'results' / host / 'test/cuda124'
            check = json.loads((out / 'run_checks.json').read_text())
            self.assertEqual(check['expected_training_runs'], 22)
            self.assertTrue(check['passed'])
            self.assertEqual((out / 'status.txt').read_text().strip(), 'COMPLETE')
            self.assertTrue((self.root / 'work/reruns' / host / 'test/cuda124/package_gpu1/repro_torch/mock.so').is_file())
        self.assertEqual((self.source / 'repro_torch/repro_torch/old_backend.so').read_text(), 'untouched')
        self.assertNotEqual(self.run_script('--configs', 'cuda124', '--gpus', '0').returncode, 0)

    def test_a100_protocol_and_failure_reporting(self):
        r = self.run_script('--configs', 'cuda124', '--gpus', '0', '--suite', 'a100')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = self.root / 'results/cuda2/test/cuda124'
        check = json.loads((out / 'run_checks.json').read_text())
        self.assertEqual(check['expected_training_runs'], 40)
        r = self.run_script('--configs', 'cuda124', '--gpus', '0', '--suite', 'a100',
                            RUN_ID='failure', FAIL_MODE='repdl')
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        failed = self.root / 'results/cuda2/failure/cuda124'
        self.assertEqual((failed / 'status.txt').read_text().strip(), 'FAILED')
        self.assertFalse(json.loads((failed / 'run_checks.json').read_text())['passed'])

    def test_snapshot_manifest_and_path_traversal(self):
        archive = self.root / 'source.tar'
        with tarfile.open(archive, 'w') as t:
            t.add(self.source, arcname='.')
        snapshot = self.root / 'snapshot'
        manifest = self.root / 'manifest.json'
        support.snapshot(archive, snapshot, manifest)
        m = json.loads(manifest.read_text())
        self.assertEqual(len(m['explicit_atomic_call_lines']), 2)
        self.assertFalse((snapshot / 'repro_torch/repro_torch/old_backend.so').exists())
        malicious = self.root / 'unsafe.tar'
        with tarfile.open(malicious, 'w') as t:
            for name in ['repro_torch/setup.py', '../escape.py']:
                info = tarfile.TarInfo(name); info.size = 1
                t.addfile(info, io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError, 'Unsafe archive path'):
            support.snapshot(malicious, self.root / 'unsafe', manifest)
        self.assertFalse((self.root / 'escape.py').exists())


if __name__ == '__main__':
    unittest.main()
