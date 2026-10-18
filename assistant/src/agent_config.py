"""Operator-owned worker settings. GPU IDs and paths never come from an LLM."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerSpec:
    gpu: str
    port: int
    cpu_moe_layers: int = 24
    threads: int = 6
    context_size: int = 16384
    moe_cache_mib: int = 0


@dataclass(frozen=True)
class AgentConfig:
    model: Path
    workers: tuple[WorkerSpec, ...]
    max_agents: int = 4
    max_iterations: int = 16
    max_tokens: int = 4196
    startup_timeout: int = 300
    request_timeout: int = 300
    batch_timeout: int = 1800
    max_transcript_bytes: int = 2_000_000

    @classmethod
    def load(cls) -> "AgentConfig":
        path = Path(os.environ.get("ASSISTANT_AGENT_CONFIG",
                                   str(Path(__file__).with_name("agents.json"))))
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("agents.json must contain an object")
        model_value = os.environ.get("ASSISTANT_WORKER_MODEL", raw.get("model", ""))
        if not isinstance(model_value, str) or not model_value.strip():
            raise ValueError("Set ASSISTANT_WORKER_MODEL to your worker GGUF path")
        model = Path(model_value).expanduser().resolve()
        if not model.is_file():
            raise ValueError("Worker GGUF is missing; configure ASSISTANT_WORKER_MODEL")
        entries = raw.get("workers")
        if not isinstance(entries, list) or not 1 <= len(entries) <= 4:
            raise ValueError("Configure one to four workers")
        workers = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each worker must be an object")
            spec = WorkerSpec(**entry)
            if not isinstance(spec.gpu, str) or not spec.gpu or "," in spec.gpu:
                raise ValueError("Each worker needs one CUDA index or GPU UUID string")
            if not (spec.gpu.isdecimal() or spec.gpu.startswith("GPU-")):
                raise ValueError("Use a CUDA index or GPU UUID")
            for name, minimum, maximum in (("port", 1024, 65535),
                    ("cpu_moe_layers", 0, 40), ("threads", 1, 48),
                    ("context_size", 4096, 262144), ("moe_cache_mib", 0, 16384)):
                value = getattr(spec, name)
                if type(value) is not int or not minimum <= value <= maximum:
                    raise ValueError(f"Invalid worker {name}")
            workers.append(spec)
        if len({s.gpu for s in workers}) != len(workers):
            raise ValueError("Workers must use different GPUs")
        if len({s.port for s in workers}) != len(workers):
            raise ValueError("Workers must use different ports")
        options = {k: v for k, v in raw.items() if k not in {"model", "workers"}}
        config = cls(model=model, workers=tuple(workers), **options)
        for name, minimum, maximum in (("max_agents", 1, len(workers)),
                ("max_iterations", 1, 64), ("max_tokens", 256, 8192),
                ("startup_timeout", 1, 1800), ("request_timeout", 1, 1800),
                ("batch_timeout", 1, 7200), ("max_transcript_bytes", 16384, 8_000_000)):
            value = getattr(config, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Invalid {name}")
        if any(config.max_tokens + 1024 >= s.context_size for s in workers):
            raise ValueError("Worker context must leave room for the prompt")
        return config


def validate_tasks(arguments: dict, maximum: int) -> list[str]:
    if not isinstance(arguments, dict) or set(arguments) - {"number_of_agents", "tasks", "summary"}:
        raise ValueError("Expected number_of_agents and tasks")
    count = arguments.get("number_of_agents")
    if type(count) is not int or not 1 <= count <= maximum:
        raise ValueError(f"number_of_agents must be between 1 and {maximum}")
    tasks = arguments.get("tasks")
    names = [f"agent{i}_task" for i in range(1, count + 1)]
    if not isinstance(tasks, dict) or set(tasks) != set(names):
        raise ValueError("tasks must contain exactly: " + ", ".join(names))
    if any(not isinstance(tasks[n], str) or not tasks[n].strip()
           or len(tasks[n]) > 16000 for n in names):
        raise ValueError("Each task must be a nonempty string of at most 16000 characters")
    return [tasks[n].strip() for n in names]


DELEGATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_agent",
        "description": (
            "Run independent workers concurrently, each on its own GPU and copied sandbox. "
            "Pass number_of_agents and tasks with exactly agent1_task through agentN_task. "
            "Include all requirements and relevant context in each task; workers do not inherit "
            "the chat and cannot delegate, call the coder, access memory, or browse the web. "
            "Workers can execute sandbox commands and install approved offline packages. "
            "Their files are returned under a unique agent-results directory in your sandbox; "
            "inspect and integrate them yourself. Your original files are not auto-merged. "
            "After results, review the evidence and use call_coder yourself if necessary. "
            "The configured backend limit may be lower than four."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "number_of_agents": {"type": "integer", "minimum": 1, "maximum": 4},
                "tasks": {"type": "object", "properties": {
                    f"agent{i}_task": {"type": "string", "minLength": 1, "maxLength": 16000}
                    for i in range(1, 5)}, "additionalProperties": False},
                "summary": {"type": "string"},
            },
            "required": ["number_of_agents", "tasks"],
            "additionalProperties": False,
        },
    },
}
