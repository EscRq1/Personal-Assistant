import ast
import json
from pathlib import Path
from queue import Queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import user_approvals as ua
import package_requests as pr
from desktop_manager import DesktopManager


def context():
    return SimpleNamespace(cancel_event=threading.Event(), stream_queue=Queue(),
                           desktop_access=False, sandbox_context_id='draft-one', chat_id=None)


def start_request(broker, ctx, kind='desktop'):
    result = []
    thread = threading.Thread(target=lambda: result.append(broker.request(ctx, kind, {'reason': 'test'})))
    thread.start()
    packet = ctx.stream_queue.get(timeout=2)[1]['approval_request']
    return thread, result, packet


class ApprovalTests(unittest.TestCase):
    def test_allow_is_one_use_and_capability_protected(self):
        broker, ctx = ua.ApprovalBroker(), context()
        thread, result, packet = start_request(broker, ctx)
        self.assertFalse(broker.decide(packet['id'], 'wrong', True))
        self.assertFalse(broker.decide(packet['id'], packet['token'], 'true'))
        self.assertTrue(broker.decide(packet['id'], packet['token'], True))
        self.assertFalse(broker.decide(packet['id'], packet['token'], True))
        thread.join(2)
        self.assertEqual(result, ['approved'])
        self.assertEqual(broker.pending, {})
        self.assertNotIn('token', ctx.stream_queue.get(timeout=2)[1]['approval_resolved'])

    def test_denial_and_cancellation(self):
        for deny in (True, False):
            broker, ctx = ua.ApprovalBroker(), context()
            thread, result, packet = start_request(broker, ctx)
            if deny:
                broker.decide(packet['id'], packet['token'], False)
            else:
                ctx.cancel_event.set()
                self.assertFalse(broker.decide(packet['id'], packet['token'], True))
            thread.join(2)
            self.assertEqual(result, ['denied' if deny else 'cancelled'])
            self.assertEqual(broker.pending, {})

    def test_expiry(self):
        broker, ctx = ua.ApprovalBroker(), context()
        with patch.object(ua, 'APPROVAL_SECONDS', .05):
            thread, result, packet = start_request(broker, ctx)
            thread.join(2)
        self.assertEqual(result, ['expired'])
        self.assertFalse(broker.decide(packet['id'], packet['token'], True))

    def test_requests_have_distinct_capabilities(self):
        broker = ua.ApprovalBroker()
        first = start_request(broker, context())
        second = start_request(broker, context())
        self.assertFalse(broker.decide(first[2]['id'], second[2]['token'], True))
        for thread, result, packet in (first, second):
            broker.decide(packet['id'], packet['token'], False)
            thread.join(2)

    def test_desktop_transition_and_turn_scope(self):
        ctx = context()
        for status in ('denied', 'approved'):
            with patch.object(ua.approvals, 'request', return_value=status):
                result = ua.ask_desktop_access(ctx, 'Read a window')
            self.assertEqual(result['status'], status)
            self.assertEqual(ua.has_desktop_access(ctx), status == 'approved')
        self.assertFalse(ua.has_desktop_access(context()))
        ctx.cancel_event.set()
        self.assertFalse(ua.has_desktop_access(ctx))

    def test_schemas_follow_permission(self):
        tools = [{'function': {'name': n}} for n in
                 ('ask_desktop_access', 'desktop_act', 'desktop_observe', 'request_package')]
        ctx = context()
        names = lambda: {x['function']['name'] for x in ua.filter_desktop_tools(tools, ctx)}
        self.assertEqual(names(), {'ask_desktop_access', 'request_package'})
        ctx.desktop_access = True
        self.assertEqual(names(), {'desktop_act', 'desktop_observe', 'request_package'})
        ctx.cancel_event.set()
        self.assertEqual(names(), {'ask_desktop_access', 'request_package'})

    def test_hidden_desktop_calls_are_rejected_before_worker_start(self):
        manager = DesktopManager()
        with patch.object(manager, 'start') as start:
            for command in ('observe', 'act'):
                self.assertEqual(manager.call(command, context())['error'], 'desktop_access_required')
            start.assert_not_called()
        manager.close()

    def test_controller_filters_actual_tool_payload(self):
        # Load the real controller class without loading models/RAG/GPU dependencies.
        source = Path(__file__).resolve().parents[1] / 'assistant.py'
        tree = ast.parse(source.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ToolController')
        ctx = context()
        namespace = {'filter_desktop_tools': ua.filter_desktop_tools,
                     '_execution_context': SimpleNamespace(get=lambda: ctx),
                     'configured_delegate_schema': lambda: {}, 'DESKTOP_NAMES': {'desktop_act', 'desktop_observe'}}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        controller = namespace['ToolController'].__new__(namespace['ToolController'])
        controller.tools = [{'function': {'name': n}} for n in
                            ('ask_desktop_access', 'desktop_act', 'desktop_observe', 'sandbox_install', 'request_package')]
        names = lambda: {x['function']['name'] for x in controller.get_available_tools()}
        self.assertEqual(names(), {'ask_desktop_access', 'request_package'})
        ctx.desktop_access = True
        self.assertEqual(names(), {'desktop_act', 'desktop_observe', 'request_package'})
        self.assertEqual({x['function']['name'] for x in controller.get_coding_tools('coder')}, {'request_package'})


class PackageTests(unittest.TestCase):
    def test_invalid_requirements_never_prompt_or_download(self):
        for value in ('https://example.org/a.whl', 'foo @ https://example.org/a.whl', '-r file', 'foo; os_name=="posix"'):
            with patch.object(pr.approvals, 'request') as request, patch.object(pr, 'download_and_install') as install:
                self.assertIn('error', pr.request_package({'packages': [value], 'reason': 'test'}, context()))
                request.assert_not_called()
                install.assert_not_called()

    def test_denial_never_starts_installation(self):
        manager = Mock()
        manager.get_or_create_for_context.return_value = SimpleNamespace(id='session-one')
        for status in ('denied', 'cancelled', 'expired'):
            with patch('sandbox_tools.get_manager', return_value=manager), \
                 patch.object(pr.approvals, 'request', return_value=status), \
                 patch.object(pr, 'download_and_install') as install:
                result = pr.request_package({'packages': ['requests>=2'], 'reason': 'Download data'}, context())
                self.assertEqual(result['status'], status)
                install.assert_not_called()

    def test_approval_installs_exact_displayed_packages_and_session(self):
        manager, ctx = Mock(), context()
        session = SimpleNamespace(id='session-one')
        manager.get_or_create_for_context.return_value = session
        with patch('sandbox_tools.get_manager', return_value=manager), \
             patch.object(pr.approvals, 'request', return_value='approved') as request, \
             patch.object(pr, 'download_and_install', return_value={'exit_code': 0}) as install:
            result = pr.request_package({'packages': ['requests>=2'], 'reason': 'Download data'}, ctx)
            install.assert_called_once_with(manager, session, ['requests>=2'], ctx)
            self.assertEqual(request.call_args.args[2]['session_id'], session.id)
            self.assertEqual(request.call_args.args[2]['packages'], ['requests>=2'])
            self.assertEqual(result['exit_code'], 0)

    def test_cancellation_after_approval_prevents_download(self):
        manager, ctx = Mock(), context()
        def approve(*args):
            ctx.cancel_event.set()
            return 'approved'
        with patch('sandbox_tools.get_manager', return_value=manager), \
             patch.object(pr.approvals, 'request', side_effect=approve), \
             patch.object(pr, 'download_and_install') as install:
            self.assertEqual(pr.request_package({'packages': ['requests'], 'reason': 'test'}, ctx)['status'], 'cancelled')
            install.assert_not_called()

    def test_download_uses_static_script_and_installs_only_in_sandbox(self):
        manager, ctx, process = Mock(), context(), Mock()
        process.poll.return_value = 0
        process.returncode = 0
        manager.config.model_output_bytes = 1000
        manager.install_manual.return_value.to_dict.return_value = {'exit_code': 0, 'stdout': 'ok', 'stderr': ''}
        session = SimpleNamespace(id='session-one')
        with patch.object(pr.subprocess, 'Popen', return_value=process) as popen:
            result = pr.download_and_install(manager, session, ['requests>=2'], ctx)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[:2], ['bash', str(pr.SCRIPT)])
        self.assertEqual(argv[-1], 'requests>=2')
        self.assertNotIn('shell', popen.call_args.kwargs)
        self.assertEqual(manager.install_manual.call_args.args[0], session)
        self.assertEqual(result['exit_code'], 0)

    def test_download_failure_does_not_install(self):
        process, manager = Mock(), Mock()
        process.poll.return_value = 1
        process.returncode = 1
        with patch.object(pr.subprocess, 'Popen', return_value=process):
            result = pr.download_and_install(manager, SimpleNamespace(id='session-one'), ['requests'], context())
        self.assertEqual(result['error'], 'download_failed')
        manager.install_manual.assert_not_called()


class DecisionEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_ui_decisions_and_replay(self):
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        app = FastAPI()
        source = Path(__file__).resolve().parents[1] / 'main.py'
        tree = ast.parse(source.read_text())
        route = next(x for x in tree.body if isinstance(x, ast.AsyncFunctionDef) and x.name == 'approval_decision')
        exec(compile(ast.Module(body=[route], type_ignores=[]), str(source), 'exec'),
             {'app': app, 'Request': Request, 'JSONResponse': JSONResponse, 'json': json})
        ctx = context()
        thread, result, packet = start_request(ua.approvals, ctx)
        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 1234))
        url = '/approvals/' + packet['id'] + '/decision'
        body = {'token': packet['token'], 'allow': True}
        async with httpx.AsyncClient(transport=transport, base_url='http://localhost') as client:
            self.assertEqual((await client.post(url, json=body, headers={'Origin': 'https://evil.example'})).status_code, 403)
            self.assertEqual((await client.post(url, json={'token': 'wrong', 'allow': True})).status_code, 409)
            self.assertEqual((await client.post(url, json={**body, 'packages': ['changed']})).status_code, 400)
            self.assertEqual((await client.post(url, content=b'x' * 2049)).status_code, 413)
            self.assertEqual((await client.post(url, json=body, headers={'Origin': 'null'})).status_code, 403)
            self.assertEqual((await client.post(url, json=body)).status_code, 200)
            self.assertEqual((await client.post(url, json=body)).status_code, 409)
        thread.join(2)
        self.assertEqual(result, ['approved'])


if __name__ == '__main__':
    unittest.main()
