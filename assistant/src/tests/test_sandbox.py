from __future__ import annotations
import ast
import asyncio
import contextvars
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest.mock import patch

from sandbox import SandboxConfig as RealSandboxConfig, SandboxManager, ExecutionResult, SessionState
from sandbox.docker_exec import DockerExecutor
from sandbox.executor import run_bounded
from sandbox.git_baseline import init_command
from sandbox.result import BoundedOutput
from sandbox_tools import SANDBOX_EXEC_SCHEMA, sandbox_exec

ROOT = Path(__file__).resolve().parents[1]


def SandboxConfig(**kwargs):
    # Unit tests may run in a root-only CI container without Docker. Mock only
    # configuration identity validation; production still rejects root.
    uid, gid = os.getuid() or 1000, os.getgid() or 1000
    with patch('sandbox.config.os.getuid', return_value=uid), patch('sandbox.config.os.getgid', return_value=gid):
        return RealSandboxConfig(uid=uid, gid=gid, **kwargs)



class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def exec(self, session, command, timeout, cancel_event=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.calls.append((session.id, command))
        try:
            if command == 'wait':
                cancel_event.wait(2)
            else:
                time.sleep(0.04)
            return ExecutionResult('ok', '', 0, 40, cancelled=cancel_event.is_set())
        finally:
            with self.lock:
                self.active -= 1

    def reconcile(self):
        return []


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'main.py').write_text('some_symbol = 42\nprint(some_symbol)\n')
        (self.source / 'tests').mkdir()
        (self.source / 'tests/test_main.py').write_text('def test_ok():\n    assert True\n')
        self.executor = FakeExecutor()
        self.manager = SandboxManager(SandboxConfig(git_enabled=False), self.root / 'sandboxes', self.executor)

    def tearDown(self):
        if not self.manager._closed:
            self.manager.close()
        self.temp.cleanup()

    def test_copies_reset_originals_and_symlink_cleanup(self):
        s = self.manager.create_session([self.source])
        (s.workspace / 'main.py').unlink()
        (s.workspace / 'escape').symlink_to(self.source, target_is_directory=True)
        self.manager.reset(s)
        self.assertEqual((s.workspace / 'main.py').read_text(), (self.source / 'main.py').read_text())
        self.assertFalse((s.workspace / 'escape').exists())
        self.assertFalse((s.workspace / 'main.py').stat().st_ino == (self.source / 'main.py').stat().st_ino)
        self.manager.destroy(s)
        self.assertTrue((self.source / 'main.py').exists())

    def test_reject_symlinks_and_special_files_atomically(self):
        s = self.manager.create_session()
        (self.source / 'unsafe').symlink_to('/etc/passwd')
        with self.assertRaises((OSError, ValueError)):
            self.manager.import_files(s, [self.source])
        self.assertEqual(list(s.workspace.iterdir()), [])
        (self.source / 'unsafe').unlink()
        os.mkfifo(self.source / 'fifo')
        with self.assertRaises(ValueError):
            self.manager.import_files(s, [self.source])
        self.assertEqual(list(s.workspace.iterdir()), [])

    def test_reject_internal_symlink_and_symlink_ancestor(self):
        s = self.manager.create_session()
        (self.source / 'internal').symlink_to('main.py')
        with self.assertRaises(OSError):
            self.manager.import_files(s, [self.source])
        link = self.root / 'linked'
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(OSError):
            self.manager.import_files(s, [link / 'main.py'])

    def test_duplicate_import_never_follows_destination_link(self):
        s = self.manager.create_session([self.source])
        (s.workspace / 'main.py').unlink()
        (s.workspace / 'main.py').symlink_to(self.source / 'main.py')
        paths = self.manager.import_files(s, [self.source / 'main.py'])
        self.assertEqual(paths, ['/workspace/main.py.import-1'])
        self.assertEqual((self.source / 'main.py').read_text(), 'some_symbol = 42\nprint(some_symbol)\n')

    def test_budgets(self):
        self.manager.close()
        self.manager = SandboxManager(SandboxConfig(git_enabled=False, max_import_bytes=5), self.root / 'sandboxes', self.executor)
        s = self.manager.create_session()
        with self.assertRaises(ValueError):
            self.manager.import_files(s, [self.source])
        self.assertEqual(list(s.workspace.iterdir()), [])

    def test_missing_workspace_is_recoverable_by_reset(self):
        s = self.manager.create_session([self.source])
        import shutil
        shutil.rmtree(s.workspace)
        with self.assertRaises(OSError):
            self.manager.exec(s, 'ls')
        self.manager.reset(s)
        self.assertTrue((s.workspace / 'main.py').exists())

    def test_context_identity_restart_and_saved_alias(self):
        s = self.manager.get_or_create_for_context('unsaved:abc')
        self.manager.bind_context('unsaved:abc', 'saved-id')
        self.assertIs(s, self.manager.get_or_create_for_context('saved-id'))
        self.manager.close()
        self.manager = SandboxManager(SandboxConfig(git_enabled=False), self.root / 'sandboxes', self.executor)
        self.assertEqual(s.id, self.manager.get_or_create_for_context('saved-id').id)
        self.assertNotEqual(s.id, self.manager.get_or_create_for_context('other-id').id)

    def test_reject_second_manager(self):
        with self.assertRaises(RuntimeError):
            SandboxManager(SandboxConfig(git_enabled=False), self.root / 'sandboxes')

    def test_session_paths_not_arbitrary(self):
        for invalid in ('../../etc', '/etc', '', 'x' * 32):
            with self.assertRaises(ValueError):
                self.manager.get_session(invalid)

    def test_same_session_serialized(self):
        s = self.manager.create_session()
        workers = [threading.Thread(target=self.manager.exec, args=(s, 'true')) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(3)
            self.assertFalse(worker.is_alive())
        self.assertEqual(self.executor.max_active, 1)

    def test_different_sessions_concurrent(self):
        sessions = [self.manager.create_session(), self.manager.create_session()]
        workers = [threading.Thread(target=self.manager.exec, args=(s, 'wait')) for s in sessions]
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 1
        while self.executor.active < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.executor.active, 2)
        for s in sessions:
            self.manager.cancel(s)
        for worker in workers:
            worker.join(3)

    def test_cancel_retains_workspace(self):
        s = self.manager.create_session([self.source])
        results = []
        worker = threading.Thread(target=lambda: results.append(self.manager.exec(s, 'wait')))
        worker.start()
        while not self.executor.active:
            time.sleep(.01)
        self.manager.cancel(s)
        worker.join(3)
        self.assertTrue(results[0].cancelled)
        self.assertTrue((s.workspace / 'main.py').exists())
        self.assertEqual(self.manager.exec(s, 'true').exit_code, 0)

    def test_waiter_cancel_does_not_cancel_owner(self):
        s = self.manager.create_session()
        owner_event, waiting_event = threading.Event(), threading.Event()
        worker = threading.Thread(target=self.manager.exec, args=(s, 'wait'), kwargs={'cancel_event': owner_event})
        worker.start()
        while not self.executor.active:
            time.sleep(.01)
        waiting_event.set()
        result = self.manager.exec(s, 'never', cancel_event=waiting_event)
        self.assertTrue(result.cancelled)
        self.assertFalse(owner_event.is_set())
        owner_event.set()
        worker.join(3)

    def test_cleanup_failure_quarantines(self):
        s = self.manager.create_session()
        with patch.object(self.executor, 'exec', return_value=ExecutionResult('', '', None, 0, error='cleanup_failed')):
            self.manager.exec(s, 'true')
        self.assertEqual(s.state, SessionState.BROKEN)
        for op in (lambda: self.manager.exec(s, 'true'), lambda: self.manager.destroy(s), lambda: self.manager.reset(s)):
            with self.assertRaises(RuntimeError):
                op()
        self.manager.reconcile()
        self.assertEqual(s.state, SessionState.READY)

    def test_reset_repairs_model_chmod_without_following_links(self):
        s = self.manager.create_session([self.source])
        (s.workspace/'tests').chmod(0)
        s.workspace.chmod(0)
        self.manager.reset(s)
        self.assertTrue((s.workspace/'tests/test_main.py').exists())
        self.assertEqual((s.workspace/'main.py').read_text(), (self.source/'main.py').read_text())

    def test_audit_failure_nonfatal(self):
        s = self.manager.create_session()
        with patch('sandbox.manager.sqlite3.connect', side_effect=OSError('test failure')):
            with self.assertLogs('sandbox.manager', level='ERROR'):
                self.assertEqual(self.manager.exec(s, 'true').exit_code, 0)

    def test_model_cannot_import_or_choose_session(self):
        class Context:
            cancel_event = threading.Event()
            chat_id = 'chat'
            sandbox_context_id = None
        self.assertEqual(set(SANDBOX_EXEC_SCHEMA['function']['parameters']['properties']), {'command', 'timeout'})
        for key in ('host_paths', 'approved_paths', 'session_id', 'container_id', 'workspace'):
            with patch('sandbox_tools.get_manager', side_effect=AssertionError('must not create')):
                result = sandbox_exec({'command': 'ls', key: '/etc'}, Context())
                self.assertEqual(result['error'], 'invalid_arguments')
        with patch('sandbox_tools.get_manager', return_value=self.manager):
            for timeout in (False, 0, -1, 301, 1.5):
                self.assertEqual(sandbox_exec({'command': 'ls', 'timeout': timeout}, Context())['error'], 'invalid_arguments')
            self.assertEqual(sandbox_exec({'command': 'ls'}, Context())['stdout'], 'ok')


class TransportTests(unittest.TestCase):
    def test_flood_stdout_and_stderr_have_bounded_parent_allocations(self):
        script = "import os\nfor _ in range(4096):\n os.write(1,b'x'*16384); os.write(2,b'y'*16384)"
        tracemalloc.start()
        result = run_bounded([sys.executable, '-c', script], timeout=20, stdout_limit=4096, stderr_limit=2048)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.stdout_truncated and result.stderr_truncated)
        self.assertLess(len(result.stdout), 4200)
        self.assertLess(len(result.stderr), 2150)
        self.assertLess(peak, 2_000_000)

    def test_timeout(self):
        result = run_bounded([sys.executable, '-c', 'while True: pass'], timeout=.15, stdout_limit=100, stderr_limit=100)
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)

    def test_cancellation(self):
        event = threading.Event()
        timer = threading.Timer(.1, event.set)
        timer.start()
        result = run_bounded([sys.executable, '-c', 'while True: pass'], timeout=10,
                             stdout_limit=100, stderr_limit=100, cancel_event=event)
        timer.join()
        self.assertTrue(result.cancelled)
        self.assertFalse(result.timed_out)

    def test_nonzero_stderr(self):
        result = run_bounded([sys.executable, '-c', 'import sys;sys.stderr.write("bad");sys.exit(7)'],
                             timeout=5, stdout_limit=100, stderr_limit=100)
        self.assertEqual((result.exit_code, result.stderr), (7, 'bad'))

    def test_zero_cap_and_utf8(self):
        buffer = BoundedOutput(3)
        buffer.feed('🙂'.encode())
        buffer.feed(b'x' * 10000)
        self.assertEqual(len(buffer.data), 3)
        self.assertIn('[OUTPUT TRUNCATED]', buffer.text())

    def test_defensive_flags_only_exact_workspace_mount(self):
        from sandbox.session import SandboxSession
        config = SandboxConfig()
        executor = DockerExecutor(config, Path('/tmp/private'))
        session = SandboxSession('a' * 32, Path('/tmp/private/a/workspace'))
        args = executor.create_args(session, 'assistant-sandbox-exec-test', 'rm -rf /')
        for flag in ('--read-only', '--cap-drop', '--security-opt', '--pids-limit', '--memory', '--memory-swap', '--cpus', '--tmpfs'):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index('--network') + 1], 'none')
        self.assertEqual(args[args.index('--user') + 1], f'{config.uid}:{config.gid}')
        self.assertEqual(args.count('--mount'), 1)
        self.assertIn('src=/tmp/private/a/workspace', args[args.index('--mount') + 1])
        self.assertIn('bind-recursive=disabled', args[args.index('--mount') + 1])
        self.assertNotIn('/var/run/docker.sock', ' '.join(args))
        self.assertEqual(args[-1], 'rm -rf /')
        self.assertEqual(args[args.index('--log-driver') + 1], 'none')

    def test_git_script_compiles(self):
        script = shlex.split(init_command(SandboxConfig()))[-1]
        compile(script, '<container-git>', 'exec')

    def test_git_policy_and_incremental_import_on_trusted_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'main.py').write_text('old\n')
            (root / 'weights.gguf').write_text('untracked')
            (root / 'huge.py').write_text('x' * 2_100_000)
            def init(imported=None):
                script = shlex.split(init_command(SandboxConfig(), imported))[-1]
                script = script.replace("pathlib.Path('/workspace')", 'pathlib.Path(' + repr(str(root)) + ')')
                result = run_bounded([sys.executable, '-I', '-c', script], timeout=10,
                                     stdout_limit=4096, stderr_limit=4096)
                self.assertEqual(result.exit_code, 0, result.stderr)
            init()
            tracked = subprocess.check_output(['git','ls-files'], cwd=root).decode()
            self.assertIn('main.py', tracked)
            self.assertNotIn('weights.gguf', tracked)
            self.assertNotIn('huge.py', tracked)
            (root / 'main.py').write_text('edited\n')
            subprocess.run(['git','add','main.py'],cwd=root,check=True)
            (root / 'extra.py').write_text('new\n')
            init(['extra.py'])
            baseline = subprocess.check_output(['git','show','HEAD:main.py'],cwd=root).decode()
            self.assertEqual(baseline, 'old\n')
            staged = subprocess.check_output(['git','diff','--cached'],cwd=root).decode()
            self.assertIn('+edited', staged)
            init(['weights.gguf'])

    def test_production_config_rejects_root(self):
        with patch('sandbox.config.os.getuid', return_value=0), patch('sandbox.config.os.getgid', return_value=0):
            with self.assertRaises(ValueError):
                RealSandboxConfig(uid=0, gid=0)

    def test_uncertain_create_and_failed_reconciliation_fail_closed(self):
        from sandbox.session import SandboxSession
        executor = DockerExecutor(SandboxConfig(), Path('/tmp/private'))
        session = SandboxSession('a'*32, Path('/tmp/private/a/workspace'))
        with patch('sandbox.docker_exec.run_bounded', return_value=ExecutionResult('','',None,1,timed_out=True)), \
             patch.object(executor, '_remove', return_value=True), self.assertLogs('sandbox.docker_exec', level='ERROR'):
            self.assertEqual(executor.exec(session, 'true', 1).error, 'cleanup_failed')
        with patch.object(executor, '_control', return_value=ExecutionResult('assistant-sandbox-exec-stale','',0,0)), \
             patch.object(executor, '_remove', return_value=False):
            with self.assertRaises(RuntimeError):
                executor.reconcile()

    def test_exact_container_cleanup_on_all_paths(self):
        from sandbox.session import SandboxSession
        config = SandboxConfig()
        executor = DockerExecutor(config, Path('/tmp/private'))
        session = SandboxSession('a' * 32, Path('/tmp/private/a/workspace'))
        for ended in (ExecutionResult('ok','',0,0), ExecutionResult('','',None,0,timed_out=True),
                      ExecutionResult('','',None,0,cancelled=True)):
            with patch('sandbox.docker_exec.run_bounded', side_effect=[ExecutionResult('id','',0,0), ended]), \
                 patch.object(executor, '_control', return_value=ExecutionResult('exited 7','',0,0)), \
                 patch.object(executor, '_remove', return_value=True) as remove:
                result = executor.exec(session, 'any shell', 5)
                self.assertEqual(remove.call_count, 1)
                self.assertTrue(remove.call_args.args[0].startswith('assistant-sandbox-exec-'))
                if not result.cancelled and not result.timed_out:
                    self.assertEqual(result.exit_code, 7)


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_import_before_first_prompt_reserves_unsaved_chat_identity(self):
        tree = ast.parse((ROOT/'main.py').read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == 'import_sandbox_files')
        captured = []
        class Manager:
            def get_or_create_for_context(self, context):
                captured.append(context)
                return 'session'
            def import_files(self, session, paths):
                return ['/workspace/source.py']
        namespace = {'asyncio': asyncio, 'Path': Path, 'get_sandbox_manager': lambda: Manager()}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'main.py','exec'),namespace)
        state = type('State', (), {'generation_lock': threading.Lock(), 'chat_initialized': False,
                                  'current_chat_id': None, 'sandbox_context_id': 'unsaved:stable'})()
        result = await namespace['import_sandbox_files'](state, [Path('/approved/source.py')])
        self.assertTrue(state.chat_initialized)
        self.assertEqual(captured, ['unsaved:stable'])
        self.assertEqual(result, ['/workspace/source.py'])

    async def test_real_controller_registration_context_and_async_dispatch(self):
        # Extract actual methods without starting the user's llama router or
        # importing unavailable GPU/RAG dependencies. No substitute controller.
        tree = ast.parse((ROOT / 'assistant.py').read_text())
        controller = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Controller')
        methods = [n for n in controller.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                   n.name in ('sandbox_exec', '_run_blocking_cancellable', '_consume_background_task')]
        context = contextvars.ContextVar('test_context')
        called = []
        def execute(args, ctx):
            time.sleep(.1)
            called.append((args, ctx))
            return {'stdout': 'ok'}
        namespace = {'asyncio': asyncio, '_execution_context': context, 'execute_sandbox_tool': execute}
        subset = ast.Module(body=[ast.ClassDef(name='Controller', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
        exec(compile(ast.fix_missing_locations(subset), 'assistant.py', 'exec'), namespace)
        ctx = type('Context', (), {'cancel_event': threading.Event()})()
        token = context.set(ctx)
        try:
            obj = namespace['Controller']()
            task = asyncio.create_task(obj._run_blocking_cancellable(ctx, obj.sandbox_exec, command='ls'))
            await asyncio.sleep(.02)
            self.assertFalse(task.done())  # event loop still responds during blocking work
            self.assertEqual(await task, {'stdout': 'ok'})
            self.assertIs(called[0][1], ctx)
        finally:
            context.reset(token)
        registry_spec = importlib.util.spec_from_file_location('test_registry', ROOT/'tools/tool_registry.py')
        registry = importlib.util.module_from_spec(registry_spec)
        registry_spec.loader.exec_module(registry)
        registry.register_tool('sandbox_exec', obj.sandbox_exec)
        self.assertIn(SANDBOX_EXEC_SCHEMA, registry.get_registered_tools())
        source = (ROOT/'assistant.py').read_text()
        self.assertIn('register_tool("sandbox_exec", self.sandbox_exec)', source)
        self.assertNotIn('sandbox_create', source)
        self.assertIn('ctx.sandbox_context_id = ctx.chat_id or assistant.sandbox_context_id', (ROOT/'main.py').read_text())


if __name__ == '__main__':
    unittest.main()
