"""Operator settings for conversations on one existing llama.cpp model.

Model paths, GPU allocation, KV allocation, and MTP belong to models.ini.
agents.json controls slots, per-role inference policy, and task budgets.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path

from agent_protocol import CHECK_SCHEMA


def object_values(raw, cls, label):
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain an object")
    unknown = set(raw) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {label} settings: {', '.join(sorted(unknown))}. "
                         "Model/GPU/MTP settings belong to models.ini; only the shared agent architecture is supported.")
    return dict(raw)


def integers(config, names, label):
    for name in names:
        if type(getattr(config, name)) is not int:
            raise ValueError(f"{label}.{name} must be an integer")


def inference_policy(config, label):
    if type(config.enable_thinking) is not bool:
        raise ValueError(f"{label}.enable_thinking must be a boolean")
    if not isinstance(config.reasoning_effort, str) or config.reasoning_effort not in {"low", "medium", "xhigh"}:
        raise ValueError(f"{label}.reasoning_effort must be low, medium, or xhigh")


@dataclass(frozen=True)
class WorkerSpec:
    slot: int
    context_size: int = 65536

    @classmethod
    def load(cls, raw):
        value = cls(**object_values(raw, cls, "worker"))
        validate_slot(value, "worker")
        return value


def validate_slot(spec, label):
    integers(spec, ("slot", "context_size"), label)
    if spec.slot < 0:
        raise ValueError(f"{label}.slot must be nonnegative")


@dataclass(frozen=True)
class OrchestratorConfig:
    slot: int = 0
    context_size: int = 32768
    max_iterations: int = 128
    max_tasks: int = 24
    max_tokens: int = -1
    enable_thinking: bool = True
    reasoning_effort: str = "medium"

    @classmethod
    def load(cls, raw):
        value = cls(**object_values(raw, cls, "orchestrator"))
        validate_slot(value, "orchestrator")
        integers(value, ("max_iterations", "max_tasks", "max_tokens"), "orchestrator")
        inference_policy(value, "orchestrator")
        return value


@dataclass(frozen=True)
class SharedModelConfig:
    alias: str = "assistant"
    slots: int = 4
    drain_timeout: int = 30

    @classmethod
    def load(cls, raw):
        value = cls(**object_values(raw, cls, "shared_model"))
        if not isinstance(value.alias, str) or not value.alias.strip():
            raise ValueError("shared_model.alias must be a nonempty string")
        integers(value, ("slots", "drain_timeout"), "shared_model")
        return value


@dataclass(frozen=True)
class AgentConfig:
    workers: tuple[WorkerSpec, ...]
    shared_model: SharedModelConfig = field(default_factory=SharedModelConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    max_agents: int = 3
    max_iterations: int = 64
    default_iterations: int = 16
    heartbeat_interval: int = 5
    max_tokens: int = -1
    enable_thinking: bool = True
    reasoning_effort: str = "low"
    request_timeout: int = 300
    batch_timeout: int = 1800
    max_transcript_bytes: int = 16_000_000

    @classmethod
    def load(cls):
        path = Path(os.environ.get("ASSISTANT_AGENT_CONFIG", str(Path(__file__).with_name("agents.json"))))
        values = object_values(json.loads(path.read_text()), cls, "agents")
        entries = values.get("workers")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Configure a nonempty workers array")
        values["workers"] = tuple(WorkerSpec.load(entry) for entry in entries)
        values["orchestrator"] = OrchestratorConfig.load(values.get("orchestrator", {}))
        values["shared_model"] = SharedModelConfig.load(values.get("shared_model", {}))
        config = cls(**values)
        # Operator numeric values are preserved, including max_tokens=-1.
        # Structural slot constraints ensure conversations cannot collide.
        integers(config, ("max_agents", "max_iterations", "default_iterations", "heartbeat_interval",
                         "max_tokens", "request_timeout", "batch_timeout", "max_transcript_bytes"), "agents")
        inference_policy(config, "agents")
        slots = [spec.slot for spec in config.workers] + [config.orchestrator.slot]
        if len(set(slots)) != len(slots):
            raise ValueError("Coordinator and workers must use distinct slots")
        if any(slot >= config.shared_model.slots for slot in slots):
            raise ValueError("Agent slot IDs must fit shared_model.slots")
        return config


DELEGATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_to_subagents",
        "description": (
            "Hand a complete mission to the coordinator on the existing shared assistant model. "
            "Include all requirements, relevant context, and optional preflight and acceptance checks. "
            "The coordinator decides how to decompose the mission and how many available workers "
            "to use concurrently, within the operator-configured capacity. It selects iteration "
            "budgets, reassigns idle workers, reviews results, integrates a separate candidate, "
            "and verifies the deliverable. Do not prescribe worker counts or individual assignments. "
            "Model weights remain loaded while you wait. Resume as final judge: inspect "
            "candidate_workspace, verification, reports, and blockers before adopting the candidate. "
            "Original files are not auto-overwritten. Agents cannot call the coder, browse, "
            "or access memory. Call one control tool at a time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 16000,
                },
                "preflight": {
                    "type": "array",
                    "maxItems": 8,
                    "items": CHECK_SCHEMA,
                },
                "acceptance": {
                    "type": "array",
                    "maxItems": 8,
                    "items": CHECK_SCHEMA,
                },
                "summary": {"type": "string"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
}

def configured_delegate_schema():
    """The main assistant submits a mission; worker capacity is operator-owned."""
    return deepcopy(DELEGATE_SCHEMA)