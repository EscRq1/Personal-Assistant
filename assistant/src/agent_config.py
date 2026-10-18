"""Operator-owned worker settings. GPU IDs and paths never come from an LLM."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
from agent_protocol import Assignment, TASK_SCHEMA, assignment


@dataclass(frozen=True)
class MtpConfig:
    """Operator-owned native MTP settings, copied into each worker process."""
    enabled: bool = False
    model: Path | None = None
    gpu: str | None = None
    draft_max: int = 3
    draft_min: int = 0
    gpu_layers: int = 99
    threads: int = 6
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"

    @classmethod
    def load(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("mtp must contain an object")
        values = dict(raw)
        model = values.get("model")
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise ValueError("mtp.model must be a GGUF path or null")
            values["model"] = Path(model).expanduser().resolve()
        config = cls(**values)
        if type(config.enabled) is not bool:
            raise ValueError("mtp.enabled must be a boolean")
        if config.gpu is not None:
            validate_gpu(config.gpu)
        for name in ("draft_max", "draft_min", "gpu_layers", "threads"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"mtp.{name} must be an integer")
        for name in ("cache_type_k", "cache_type_v"):
            if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
                raise ValueError(f"mtp.{name} must be a nonempty string")
        if config.enabled and config.model is not None and not config.model.is_file():
            raise ValueError("MTP head GGUF is missing; configure mtp.model")
        return config


def validate_gpu(value):
    if (not isinstance(value, str) or not value or "," in value or
            not (value.isdecimal() or value.startswith("GPU-"))):
        raise ValueError("Use one CUDA index or GPU UUID string")


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
    max_agents: int = 3
    max_iterations: int = 64
    default_iterations: int = 16
    heartbeat_interval: int = 5
    max_tokens: int = 2048
    startup_timeout: int = 300
    request_timeout: int = 300
    batch_timeout: int = 1800
    max_transcript_bytes: int = 16_000_000
    mtp: MtpConfig = MtpConfig()

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
        if not isinstance(entries, list) or not entries:
            raise ValueError("Configure a nonempty workers array")
        workers = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each worker must be an object")
            spec = WorkerSpec(**entry)
            validate_gpu(spec.gpu)
            for name in ("port", "cpu_moe_layers", "threads", "context_size", "moe_cache_mib"):
                if type(getattr(spec, name)) is not int:
                    raise ValueError(f"Worker {name} must be an integer")
            workers.append(spec)
        if len({s.gpu for s in workers}) != len(workers):
            raise ValueError("Workers must use different GPUs")
        if len({s.port for s in workers}) != len(workers):
            raise ValueError("Workers must use different ports")
        options = {k: v for k, v in raw.items() if k not in {"model", "workers"}}
        options["mtp"] = MtpConfig.load(options.get("mtp", {}))
        config = cls(model=model, workers=tuple(workers), **options)
        # These values are owned by the operator. Preserve them as written;
        # llama.cpp, the OS, and requests enforce their own supported semantics.
        for name in ("max_agents", "max_iterations", "default_iterations",
                     "heartbeat_interval", "max_tokens", "startup_timeout",
                     "request_timeout", "batch_timeout", "max_transcript_bytes"):
            if type(getattr(config, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if config.mtp.enabled:
            if config.mtp.model == model:
                raise ValueError("mtp.model must be an MTP-only head GGUF, not the full worker model")
            if (config.mtp.gpu is not None and config.mtp.model is None and
                    any(s.gpu != config.mtp.gpu for s in workers)):
                raise ValueError("Separate-GPU MTP requires mtp.model: embedded heads reuse the target weights")
        return config


def validate_tasks(arguments: dict, maximum: int, maximum_iterations: int = 64,
                   default_iterations: int = 16) -> list[Assignment]:
    if not isinstance(arguments, dict) or set(arguments) - {"number_of_agents", "tasks", "summary"}:
        raise ValueError("Expected number_of_agents and tasks")
    count = arguments.get("number_of_agents")
    if type(count) is not int or not 1 <= count <= maximum:
        raise ValueError(f"number_of_agents must be between 1 and {maximum}")
    tasks = arguments.get("tasks")
    names = [f"agent{i}_task" for i in range(1, count + 1)]
    if not isinstance(tasks, dict) or set(tasks) != set(names):
        raise ValueError("tasks must contain exactly: " + ", ".join(names))
    return [assignment(tasks[n], maximum_iterations, default_iterations) for n in names]


DELEGATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_agent",
        "description": (
            "Run independent workers concurrently, each on its own GPU and copied sandbox. "
            "Pass number_of_agents and tasks with exactly agent1_task through agentN_task. "
            "Each task is an object with task and iterations; choose a separate iteration budget "
            "for each task from complexity, estimated tool actions, a final-report round, and retry "
            "headroom. Add structured preflight and acceptance command/file checks. Use resume_from "
            "with a previous batch_id/agent_id to continue its saved workspace. "
            "Include all requirements and relevant context in each task; workers do not inherit "
            "the chat and cannot delegate, call the coder, access memory, or browse the web. "
            "Workers can execute sandbox commands and install approved offline packages. "
            "Their files are returned under a unique agent-results directory in your sandbox; "
            "inspect and integrate them yourself. Your original files are not auto-merged. "
            "After results, review the evidence and use call_coder yourself if necessary. "
            "Worker count and iteration budgets follow the operator's configuration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "number_of_agents": {"type": "integer", "minimum": 1, "maximum": 3},
                "tasks": {"type": "object", "properties": {
                    f"agent{i}_task": TASK_SCHEMA
                    for i in range(1, 5)}, "additionalProperties": False},
                "summary": {"type": "string"},
            },
            "required": ["number_of_agents", "tasks"],
            "additionalProperties": False,
        },
    },
}


def configured_delegate_schema():
    """Advertise operator-selected limits, without a hidden four-agent/64-round cap."""
    schema = deepcopy(DELEGATE_SCHEMA)
    try:
        config = AgentConfig.load()
    except (OSError, ValueError, TypeError):
        # Keep the tool available so it can report a useful configuration error.
        return schema
    count = min(config.max_agents, len(config.workers))
    if count < 1:
        return schema
    parameters = schema["function"]["parameters"]["properties"]
    parameters["number_of_agents"]["maximum"] = count
    task = deepcopy(TASK_SCHEMA)
    if config.max_iterations >= 1:
        task["properties"]["iterations"]["maximum"] = config.max_iterations
    parameters["tasks"]["properties"] = {f"agent{i}_task": task for i in range(1, count + 1)}
    return schema
