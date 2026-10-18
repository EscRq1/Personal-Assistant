"""One independently owned llama-server process per worker/GPU."""
from __future__ import annotations

import os
import atexit
import signal
import socket
import subprocess
import threading
import time

import requests

_servers = set()
_servers_lock = threading.Lock()


def stop_worker_servers():
    with _servers_lock:
        servers = list(_servers)
    errors = []
    for server in servers:
        try:
            server.stop()
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError("Could not stop worker processes: " + "; ".join(errors))


atexit.register(stop_worker_servers)


class WorkerServer:
    def __init__(self, config, spec, binary, lib_dir, log_path):
        self.config, self.spec = config, spec
        self.binary, self.lib_dir, self.log_path = binary, lib_dir, log_path
        self.process = None
        self._lock = threading.Lock()
        self._stopped = False
        self._cleaned = False
        self.url = f"http://127.0.0.1:{spec.port}"
        with _servers_lock:
            _servers.add(self)

    def command(self):
        s = self.spec
        command = [self.binary, "--model", str(self.config.model), "--alias", "worker",
                "--host", "127.0.0.1", "--port", str(s.port),
                "--device", "CUDA0", "--split-mode", "none", "--gpu-layers", "99",
                "--n-cpu-moe", str(s.cpu_moe_layers),
                "--ctx-size", str(s.context_size), "--parallel", "1",
                "--threads", str(s.threads), "--threads-batch", str(s.threads),
                "--batch-size", "256", "--ubatch-size", "128",
                "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                "--flash-attn", "on", "--jinja", "--reasoning-format", "deepseek",
                "--no-context-shift"]
        if s.moe_cache_mib:
            command += ["--moe-cache-mib", str(s.moe_cache_mib)]
        return command

    def start(self, cancel):
        with self._lock:
            if self._stopped or cancel.is_set():
                raise RuntimeError("Worker startup cancelled")
            # Never adopt an unrelated listener on this configured port.
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", self.spec.port))
            env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
            env["CUDA_VISIBLE_DEVICES"] = self.spec.gpu
            env["LD_LIBRARY_PATH"] = self.lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
            with self.log_path.open("wb") as log:
                self.process = subprocess.Popen(self.command(), env=env, stdin=subprocess.DEVNULL,
                                                stdout=log, stderr=subprocess.STDOUT,
                                                start_new_session=True)
        deadline = time.monotonic() + self.config.startup_timeout
        with requests.Session() as http:
            http.trust_env = False
            while time.monotonic() < deadline:
                if cancel.is_set():
                    raise RuntimeError("Worker startup cancelled")
                if self.process.poll() is not None:
                    raise RuntimeError(f"Worker exited with code {self.process.returncode}; see server.log")
                try:
                    with http.get(self.url + "/health", timeout=1) as response:
                        if response.status_code == 200:
                            return
                except requests.RequestException:
                    pass
                cancel.wait(0.1)
        raise TimeoutError("Worker startup timed out; see server.log")

    def stop(self):
        # Mark stopped even if a cancellation races startup.
        with self._lock:
            if self._cleaned:
                return
            self._stopped = True
            if self.process is None:
                self._cleaned = True
                with _servers_lock:
                    _servers.discard(self)
                return
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=10)
            self._cleaned = True
            with _servers_lock:
                _servers.discard(self)

    def complete(self, messages, tools, cancel, max_tokens):
        payload = {"model": "worker", "messages": messages, "tools": tools,
                   "tool_choice": "auto", "parallel_tool_calls": False,
                   "stream": True, "cache_prompt": True, "temperature": 0.6,
                   "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True}}
        content, reasoning, calls = [], [], {}
        size, finish = 0, None
        import json
        with requests.Session() as http:
            http.trust_env = False
            with http.post(self.url + "/v1/chat/completions", json=payload,
                           stream=True, timeout=(5, self.config.request_timeout)) as response:
                response.raise_for_status()
                for raw in response.iter_lines():
                    if cancel.is_set():
                        raise RuntimeError("Worker cancelled")
                    size += len(raw)
                    if size > self.config.max_transcript_bytes:
                        raise ValueError("Worker response exceeded byte limit")
                    if not raw.startswith(b"data:"):
                        continue
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError("Worker inference error: " + str(event["error"]))
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    for item in delta.get("tool_calls") or []:
                        index = item.get("index", 0)
                        if type(index) is not int or not 0 <= index < 16:
                            raise ValueError("Invalid worker tool index")
                        call = calls.setdefault(index, {"id": "", "type": "function",
                                                       "function": {"name": "", "arguments": ""}})
                        if item.get("id"):
                            call["id"] = item["id"]
                        for key in ("name", "arguments"):
                            call["function"][key] += (item.get("function") or {}).get(key) or ""
        if finish is None:
            raise RuntimeError("Worker stream ended without a finish reason")
        message = {"role": "assistant", "content": "".join(content) or None}
        if any(reasoning):
            message["reasoning_content"] = "".join(reasoning)
        if calls:
            import uuid
            message["tool_calls"] = []
            for index in sorted(calls):
                call = calls[index]
                call["id"] = call["id"] or "call_" + uuid.uuid4().hex
                message["tool_calls"].append(call)
        return message, finish
