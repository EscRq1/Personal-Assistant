"""Opt-in acceptance tests on a local Linux Docker daemon and the built image.

SANDBOX_DOCKER_TESTS=1 python -m unittest discover -s tests -p 'test_sandbox*.py' -v
These tests execute destructive commands ONLY through DockerExecutor.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import time
import unittest

from sandbox import SandboxConfig, SandboxManager
from sandbox.docker_exec import LABEL


@unittest.skipUnless(os.environ.get('SANDBOX_DOCKER_TESTS') == '1', 'Set SANDBOX_DOCKER_TESTS=1 for real Docker acceptance tests')
class DockerAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='sandbox-acceptance-')
        self.root = Path(self.temp.name)
        self.source = self.root/'original'
        self.source.mkdir()
        (self.source/'main.py').write_text('some_symbol = 42\nprint(some_symbol)\n')
        (self.source/'tests').mkdir()
        (self.source/'tests/test_main.py').write_text('import runpy\ndef test_main():\n    assert runpy.run_path("main.py")["some_symbol"] == 42\n')
        config = SandboxConfig(image=os.environ.get('SANDBOX_IMAGE', 'assistant-sandbox:latest'),
                               max_stdout_bytes=8192, max_stderr_bytes=4096)
        self.manager = SandboxManager(config, self.root/'sandboxes')
        self.session = self.manager.create_session([self.source])

    def tearDown(self):
        try:
            self.manager.reconcile()
            self.manager.destroy(self.session)
        finally:
            self.manager.close()
            self.temp.cleanup()

    def run_command(self, command, timeout=None):
        return self.manager.exec(self.session, command, timeout)

    def assert_no_containers(self):
        result = subprocess.run(['docker','ps','-aq','--filter',f'label={LABEL}={self.manager.executor.owner}'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=True)
        self.assertEqual(result.stdout.strip(), b'')

    def test_functional_sequence_and_baseline(self):
        for command, expected in [('find . -type f','main.py'), ('grep -R "some_symbol" -n .','some_symbol'),
                                  ('python main.py','42'), ('pytest -q','1 passed')]:
            result = self.run_command(command)
            self.assertEqual(result.exit_code, 0, result)
            self.assertIn(expected, result.stdout)
        self.assertEqual(self.run_command('rm main.py').exit_code, 0)
        self.assertFalse((self.session.workspace/'main.py').exists())
        self.assertTrue((self.source/'main.py').exists())
        self.manager.reset(self.session)
        self.assertTrue((self.session.workspace/'main.py').exists())
        self.assert_no_containers()

    def test_uid_root_readonly_host_boundary_network(self):
        host_secret = self.root/'HOST_SENTINEL'
        host_secret.write_text('private')
        code = '''import os, pathlib, socket
assert os.geteuid() != 0
assert not pathlib.Path(HOST_SECRET).exists()
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not pathlib.Path('/workspace/../' + HOST_SECRET.lstrip('/')).exists()
assert not pathlib.Path('/workspace/../../' + HOST_SECRET.lstrip('/')).exists()
pathlib.Path('/workspace/writable').write_text('ok')
pathlib.Path('/tmp/writable').write_text('ok')
try:
    pathlib.Path('/etc/SANDBOX_WRITE_TEST').write_text('bad')
except OSError:
    pass
else:
    raise AssertionError('root filesystem writable')
s = socket.socket(); s.settimeout(1)
try:
    s.connect(('1.1.1.1', 53))
except OSError:
    pass
else:
    raise AssertionError('network available')
print('boundary ok')
'''
        code = 'HOST_SECRET=' + repr(str(host_secret)) + '\n' + code
        result = self.run_command('python -c ' + shlex.quote(code))
        self.assertEqual(result.exit_code, 0, result)
        self.assertIn('boundary ok', result.stdout)
        self.assertEqual(host_secret.read_text(), 'private')
        self.assert_no_containers()

    def test_live_configuration_and_other_session(self):
        other = self.manager.create_session()
        (other.workspace/'OTHER_SECRET').write_text('hidden')
        command = f'test ! -e {shlex.quote(str(other.workspace))}; touch ready; sleep 30'
        event = threading.Event()
        results = []
        worker = threading.Thread(target=lambda: results.append(self.manager.exec(self.session, command, 40, event)))
        worker.start()
        try:
            deadline = time.monotonic() + 15
            while not (self.session.workspace/'ready').exists() and time.monotonic() < deadline:
                time.sleep(.05)
            listing = subprocess.run(['docker','ps','-q','--filter',f'label={LABEL}={self.manager.executor.owner}'],
                                     stdout=subprocess.PIPE, timeout=15, check=True).stdout.decode().strip()
            self.assertTrue(listing)
            data = json.loads(subprocess.run(['docker','inspect',listing], stdout=subprocess.PIPE,
                                            timeout=15, check=True).stdout)[0]
            host = data['HostConfig']
            self.assertEqual(host['NetworkMode'], 'none')
            self.assertTrue(host['ReadonlyRootfs'])
            self.assertEqual(host['CapDrop'], ['ALL'])
            self.assertIn('no-new-privileges:true', host['SecurityOpt'])
            self.assertGreater(host['Memory'], 0)
            self.assertGreater(host['NanoCpus'], 0)
            self.assertGreater(host['PidsLimit'], 0)
            binds = [m for m in data['Mounts'] if m['Type'] == 'bind']
            self.assertEqual(len(binds), 1)
            self.assertEqual(binds[0]['Source'], str(self.session.workspace))
            self.assertEqual(binds[0]['Destination'], '/workspace')
            self.assertEqual(data['Config']['User'], f'{os.getuid()}:{os.getgid()}')
        finally:
            event.set()
            worker.join(40)
            self.manager.destroy(other)
        self.assertFalse(worker.is_alive())
        self.assert_no_containers()

    def test_timeout_kills_children_and_next_command_works(self):
        code = "import subprocess,time; subprocess.Popen(['python','-c','import time; time.sleep(300)'],start_new_session=True); time.sleep(300)"
        result = self.run_command('python -c '+shlex.quote(code), 1)
        self.assertTrue(result.timed_out, result)
        self.assert_no_containers()  # no container PID namespace/children survive
        self.assertEqual(self.run_command('echo next').stdout.strip(), 'next')
        result = self.run_command("python -c 'while True: pass'", 1)
        self.assertTrue(result.timed_out)
        self.assert_no_containers()

    def test_infinite_output_flood(self):
        code = "import os\nwhile True:\n os.write(1,b'x'*65536); os.write(2,b'y'*65536)"
        result = self.run_command('python -c '+shlex.quote(code), 2)
        self.assertTrue(result.timed_out)
        self.assertTrue(result.stdout_truncated and result.stderr_truncated)
        self.assertLess(len(result.stdout), 8300)
        self.assertLess(len(result.stderr), 4200)
        self.assert_no_containers()

    def test_cancel_preserves_workspace(self):
        worker = threading.Thread(target=lambda: self.manager.exec(self.session, 'touch started; sleep 120', 150))
        worker.start()
        deadline = time.monotonic() + 15
        while not (self.session.workspace/'started').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        self.manager.cancel(self.session)
        worker.join(40)
        self.assertFalse(worker.is_alive())
        self.assertTrue((self.session.workspace/'main.py').exists())
        self.assert_no_containers()

    def test_destructive_commands_only_destroy_copies(self):
        original = (self.source/'main.py').read_bytes()
        self.run_command('rm -rf -- /workspace/* /workspace/.[!.]* /workspace/..?*')
        self.assertEqual(list(self.session.workspace.iterdir()), [])
        self.assertEqual((self.source/'main.py').read_bytes(), original)
        self.manager.reset(self.session)
        # Explicit override exercises the actual isolation, not rm's default /
        # protection. Root is read-only and non-root; host sentinel is unmounted.
        self.run_command('rm -rf --no-preserve-root /', 5)
        self.assertEqual((self.source/'main.py').read_bytes(), original)
        self.assertTrue(Path('/bin/sh').exists())
        self.manager.reset(self.session)
        self.assertEqual(self.run_command('python main.py').exit_code, 0)
        self.assert_no_containers()

    def test_errors_and_git_data_policy(self):
        result = self.run_command('nonexistent_command_672923')
        self.assertEqual(result.exit_code, 127)
        self.assertIn('not found', result.stderr)
        (self.source/'huge.py').write_bytes(b'x' * 2_100_000)
        (self.source/'weights.gguf').write_bytes(b'weights')
        self.manager.import_files(self.session, [self.source/'huge.py', self.source/'weights.gguf'])
        self.manager.reset(self.session)
        tracked = self.run_command('git ls-files')
        self.assertIn('main.py', tracked.stdout)
        self.assertNotIn('huge.py', tracked.stdout)
        self.assertNotIn('weights.gguf', tracked.stdout)
        self.assert_no_containers()
