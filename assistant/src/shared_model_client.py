"""Independent agent HTTP streams sharing an already-loaded router model.

No subprocess, GGUF load, GPU assignment, model unload, or router termination
exists here. close() aborts only this client's socket. The resource lease verifies
server-side quiescence before another GPU resource may be selected.
"""
from __future__ import annotations

import atexit
import http.client
import json
import socket
import threading
import uuid
from urllib.parse import urlsplit
from agent_config import OrchestratorConfig

_clients = set()
_registry_lock = threading.Lock()


def stop_shared_clients():
    with _registry_lock:
        clients = list(_clients)
    for client in clients:
        client.close()


class SharedModelClient:
    def __init__(self, config, spec, log_path, *, router_url):
        self.config, self.spec, self.log_path = config, spec, log_path
        self.url = router_url.rstrip("/")
        self.model = config.shared_model.alias
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._socket = self._connection = None
        self._stopped = False
        self.failed = False
        with _registry_lock:
            _clients.add(self)

    def attach(self, cancel):
        with self._lock:
            if self._stopped or self.failed or cancel.is_set():
                raise InterruptedError("Shared client startup cancelled")
        self.log_path.write_text(f"Shared router client: {self.url}; model={self.model}; slot={self.spec.slot}\n"
                                 "No model process is started or owned by this client.\n")
        self.log_path.with_name("connection.json").write_text(json.dumps({
            "backend": "shared", "router_url": self.url, "model": self.model,
            "slot": self.spec.slot}, indent=2) + "\n")

    def close(self):
        with self._lock:
            self._stopped = True
            connection, transport = self._connection, self._socket
        if transport is not None:
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if connection is not None:
            connection.close()
        with _registry_lock:
            _clients.discard(self)

    def _messages(self, messages, max_tokens):
        """Bound old agent history, preserving initial instructions and tool pairs.

        context_size is a per-agent prompt-budget hint, not a KV allocation.
        The server's unified pool remains configured in models.ini. Full worker
        transcripts stay journaled by the runtime.
        """
        boundary = next((i for i, item in enumerate(messages) if item["role"] == "assistant"), len(messages))
        prefix, groups = messages[:boundary], []
        for item in messages[boundary:]:
            if item["role"] == "tool" and groups and groups[-1][0]["role"] == "assistant":
                groups[-1].append(item)
            else:
                groups.append([item])
        reserve = max_tokens if max_tokens > 0 else 4096
        allowance = max(8192, (self.spec.context_size - reserve) * 2)
        used, tail = len(json.dumps(prefix).encode()), []
        for group in reversed(groups):
            size = len(json.dumps(group).encode())
            if used + size > allowance:
                break
            tail.insert(0, group)
            used += size
        return prefix + [item for group in tail for item in group]

    def complete(self, messages, tools, cancel, max_tokens, on_progress=None):
        # This lock serializes one conversation only. Other clients/slots run
        # concurrently and never acquire ResourceManager.lock during inference.
        with self._request_lock:
            if cancel.is_set():
                raise InterruptedError("Shared request cancelled")
            payload = {"model": self.model, "id_slot": self.spec.slot,
                "messages": self._messages(messages, max_tokens), "tools": tools,
                "tool_choice": "auto", "parallel_tool_calls": False,
                "stream": True, "cache_prompt": True, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": True}}
            policy = self.spec if isinstance(self.spec, OrchestratorConfig) else self.config
            payload["chat_template_kwargs"] = {
                "enable_thinking": policy.enable_thinking,
                "reasoning_effort": policy.reasoning_effort}
            # Pass both the API-level value and the template kwarg so an
            # embedded Qwen template sees the explicitly selected role effort.
            payload["reasoning_effort"] = policy.reasoning_effort if policy.enable_thinking else "none"
            if isinstance(self.spec, OrchestratorConfig):
                payload["tool_choice"] = "required"
            parts = urlsplit(self.url)
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError("Router URL must be http(s)")
            connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
            connection = connection_type(parts.hostname, parts.port, timeout=5)
            # A cancellation must not cause http.client.send() to reconnect
            # after close() has closed the explicitly registered socket.
            connection.auto_open = False
            response = None
            with self._lock:
                if self._stopped or self.failed:
                    raise InterruptedError("Shared client is closed")
                self._connection = connection
            try:
                connection.connect()
                with self._lock:
                    self._socket = connection.sock
                    if self._stopped or cancel.is_set():
                        raise InterruptedError("Shared request cancelled before headers")
                    connection.sock.settimeout(self.config.request_timeout)
                path = parts.path.rstrip("/") + "/v1/chat/completions"
                connection.request("POST", path, body=json.dumps(payload).encode(), headers={
                    "Content-Type": "application/json", "Accept": "text/event-stream", "Connection": "close"})
                response = connection.getresponse()
                if response.status != 200:
                    raise RuntimeError(f"Shared model HTTP {response.status}: " + response.read(4000).decode("utf-8", "replace"))
                content, reasoning, calls = [], [], {}
                size, finish = 0, None
                while True:
                    if cancel.is_set():
                        raise InterruptedError("Shared request cancelled")
                    raw = response.readline()
                    if not raw:
                        break
                    size += len(raw)
                    if size > self.config.max_transcript_bytes:
                        raise ValueError("Shared response exceeded byte limit")
                    if not raw.startswith(b"data:"):
                        continue
                    if on_progress:
                        on_progress()
                    data = raw[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError("Shared inference error: " + str(event["error"]))
                    if not event.get("choices"):
                        continue
                    choice = event["choices"][0]
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    for item in delta.get("tool_calls") or []:
                        index = item.get("index", 0)
                        if type(index) is not int or not 0 <= index < 16:
                            raise ValueError("Invalid shared tool index")
                        call = calls.setdefault(index, {"id": "", "type": "function",
                            "function": {"name": "", "arguments": ""}})
                        if item.get("id"):
                            call["id"] = item["id"]
                        for key in ("name", "arguments"):
                            call["function"][key] += (item.get("function") or {}).get(key) or ""
                if finish is None:
                    raise RuntimeError("Shared stream ended without a finish reason")
                message = {"role": "assistant", "content": "".join(content) or None}
                if any(reasoning):
                    message["reasoning_content"] = "".join(reasoning)
                if calls:
                    message["tool_calls"] = []
                    for index in sorted(calls):
                        value = calls[index]
                        value["id"] = value["id"] or "call_" + uuid.uuid4().hex
                        message["tool_calls"].append(value)
                return message, finish
            except BaseException:
                # Do not assign fresh work to a slot whose previous stream
                # ended ambiguously. Other healthy clients can still run; the
                # lease will verify that all server-side inference has drained.
                self.failed = True
                raise
            finally:
                if response is not None:
                    response.close()
                connection.close()
                with self._lock:
                    self._connection = self._socket = None


atexit.register(stop_shared_clients)