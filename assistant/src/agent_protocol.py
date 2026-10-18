"""Validated worker assignments, completion reports, and sandbox-only checks."""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath


def workspace_path(value):
    if not isinstance(value, str) or not value or len(value) > 1024 or "\0" in value:
        raise ValueError("Artifact paths must be nonempty workspace-relative strings")
    if value.startswith("/workspace/"):
        value = value[len("/workspace/"):]
    if "\\" in value or value.startswith("/") or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError("Artifact paths must stay inside /workspace, without . or .. components")
    if PurePosixPath(value).parts[0] in {".git", ".venv", ".packages", "node_modules"}:
        raise ValueError("Runtime/cache directories cannot be artifacts")
    return value


def validate_checks(value, maximum_timeout=300):
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Checks must be an array of at most eight objects")
    checks = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Use structured command/file checks, not prose acceptance strings")
        kind = item.get("type")
        allowed = {"type", "command", "timeout"} if kind == "command" else {
            "type", "path", "min_bytes", "magic_hex", "timeout"}
        if kind not in {"command", "file"} or set(item) - allowed:
            raise ValueError("Unknown check type or fields")
        check = dict(item)
        timeout = check.setdefault("timeout", min(60, maximum_timeout))
        if type(timeout) is not int or not 1 <= timeout <= maximum_timeout:
            raise ValueError(f"Check timeout must be between 1 and {maximum_timeout}")
        if kind == "command":
            command = check.get("command")
            if not isinstance(command, str) or not command.strip() or "\0" in command or len(command) > 8000:
                raise ValueError("Check command must be a nonempty string of at most 8000 characters")
        else:
            check["path"] = workspace_path(check.get("path"))
            size = check.setdefault("min_bytes", 1)
            if type(size) is not int or not 0 <= size <= 512_000_000:
                raise ValueError("min_bytes must be a nonnegative integer up to 512000000")
            magic = check.get("magic_hex", "")
            if not isinstance(magic, str) or len(magic) > 128 or len(magic) % 2 or not re.fullmatch("[0-9a-fA-F]*", magic):
                raise ValueError("magic_hex must be an even-length hexadecimal string, at most 128 digits")
        checks.append(check)
    return tuple(checks)


@dataclass(frozen=True)
class Assignment:
    task: str
    iterations: int
    acceptance: tuple[dict, ...] = ()
    preflight: tuple[dict, ...] = ()
    resume_from: dict | None = None


def assignment(value, maximum_iterations, default_iterations):
    # Legacy string tasks remain accepted; new tool schemas ask for explicit budgets.
    if isinstance(value, str):
        value = {"task": value, "iterations": min(default_iterations, maximum_iterations)}
    if not isinstance(value, dict) or set(value) - {"task", "iterations", "acceptance", "preflight", "resume_from"}:
        raise ValueError("Each task must contain task, iterations, and optional acceptance/preflight/resume_from")
    task, count = value.get("task"), value.get("iterations")
    if not isinstance(task, str) or not task.strip() or len(task) > 16000:
        raise ValueError("task must be a nonempty string of at most 16000 characters")
    if type(count) is not int or not 1 <= count <= maximum_iterations:
        raise ValueError(f"iterations must be between 1 and the backend ceiling {maximum_iterations}")
    resume = value.get("resume_from")
    if resume is not None:
        if not isinstance(resume, dict) or set(resume) != {"batch_id", "agent_id"}:
            raise ValueError("resume_from requires batch_id and agent_id")
        if not isinstance(resume["batch_id"], str) or not re.fullmatch("[0-9a-f]{32}", resume["batch_id"]):
            raise ValueError("Invalid resume batch ID")
        if not isinstance(resume["agent_id"], str) or not re.fullmatch("agent[1-9][0-9]*", resume["agent_id"]):
            raise ValueError("Invalid resume agent ID")
    return Assignment(task.strip(), count, validate_checks(value.get("acceptance", [])),
                      validate_checks(value.get("preflight", [])), resume)


def completion(value):
    if not isinstance(value, dict) or set(value) - {"status", "result", "artifacts", "blockers", "next_step"}:
        raise ValueError("Completion must contain status, result, artifacts, blockers, next_step only")
    if set(value) != {"status", "result", "artifacts", "blockers", "next_step"}:
        raise ValueError("Completion requires status, result, artifacts, blockers, next_step")
    if value["status"] not in {"completed", "partial", "blocked"}:
        raise ValueError("Completion status must be completed, partial, or blocked")
    for key, maximum in (("result", 16000), ("next_step", 4000)):
        if not isinstance(value[key], str) or len(value[key]) > maximum:
            raise ValueError(f"Completion {key} must be a string of at most {maximum} characters")
    for key, maximum_items, maximum_length in (("artifacts", 128, 1024), ("blockers", 16, 2000)):
        items = value[key]
        if not isinstance(items, list) or len(items) > maximum_items or any(
                not isinstance(x, str) or not x.strip() or len(x) > maximum_length for x in items):
            raise ValueError(f"Invalid completion {key}")
    result = dict(value)
    result["artifacts"] = list(dict.fromkeys(workspace_path(x) for x in value["artifacts"]))
    return result


FINISH_SCHEMA = {"type": "function", "function": {
    "name": "finish_task",
    "description": "End your task with a structured report. Report partial/blocked honestly. The backend independently checks artifacts and acceptance criteria. Call this tool alone.",
    "parameters": {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["completed", "partial", "blocked"]},
        "result": {"type": "string", "maxLength": 16000},
        "artifacts": {"type": "array", "maxItems": 128, "items": {"type": "string"}},
        "blockers": {"type": "array", "maxItems": 16, "items": {"type": "string"}},
        "next_step": {"type": "string", "maxLength": 4000}},
        "required": ["status", "result", "artifacts", "blockers", "next_step"], "additionalProperties": False}}}


CHECK_SCHEMA = {"type": "object", "properties": {
    "type": {"type": "string", "enum": ["command", "file"]},
    "command": {"type": "string", "description": "For command checks: Bash command that must exit zero."},
    "path": {"type": "string", "description": "For file checks: workspace-relative regular file."},
    "min_bytes": {"type": "integer", "minimum": 0},
    "magic_hex": {"type": "string", "description": "Optional file prefix in hexadecimal; PDF is 25504446."},
    "timeout": {"type": "integer", "minimum": 1, "maximum": 300}},
    "required": ["type"], "additionalProperties": False}

TASK_SCHEMA = {"type": "object", "properties": {
    "task": {"type": "string", "minLength": 1, "maxLength": 16000},
    "iterations": {"type": "integer", "minimum": 1,
        "description": "Choose a per-agent generation budget from task complexity: estimated tool actions + final report + retry headroom. One action tool per round. Backend ceiling applies."},
    "preflight": {"type": "array", "maxItems": 8, "items": CHECK_SCHEMA},
    "acceptance": {"type": "array", "maxItems": 8, "items": CHECK_SCHEMA},
    "resume_from": {"type": "object", "properties": {
        "batch_id": {"type": "string"}, "agent_id": {"type": "string", "pattern": "^agent[1-9][0-9]*$"}},
        "required": ["batch_id", "agent_id"], "additionalProperties": False}},
    "required": ["task", "iterations"], "additionalProperties": False}


# Executed with isolated Python INSIDE the existing no-network Docker sandbox.
# Each path component is opened relative to a pinned fd, without following links.
FILE_CHECK_SCRIPT = '''import json, os, stat, sys
c=json.loads(sys.argv[1]); fd=os.open('/workspace', os.O_RDONLY|os.O_DIRECTORY)
try:
    parts=c['path'].split('/')
    for part in parts[:-1]:
        new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        os.close(fd); fd=new
    f=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    try:
        info=os.fstat(f)
        ok=stat.S_ISREG(info.st_mode) and info.st_size>=c.get('min_bytes',1)
        magic=bytes.fromhex(c.get('magic_hex',''))
        if magic: ok=ok and os.read(f,len(magic))==magic
        print(json.dumps({'path':c['path'],'size_bytes':info.st_size,'passed':ok}))
    finally: os.close(f)
except OSError as e:
    ok=False; print(json.dumps({'path':c['path'],'passed':False,'error':str(e)}))
finally: os.close(fd)
sys.exit(0 if ok else 1)
'''


def run_checks(manager, session, checks, cancel, phase):
    results = []
    for check in checks:
        record = {"check": check, "status": "skipped", "exit_code": None}
        if cancel.is_set():
            record["reason"] = "cancelled"
        else:
            command = check.get("command")
            if check["type"] == "file":
                command = "/usr/local/bin/python -I -c " + shlex.quote(FILE_CHECK_SCRIPT) + " " + shlex.quote(json.dumps(check))
            try:
                value = manager.exec(session, command, check["timeout"], cancel,
                                     caller="delegate:" + phase).to_dict()
                ok = value.get("exit_code") == 0 and not any(value.get(k) for k in ("error", "cancelled", "timed_out"))
                record.update(status="passed" if ok else "failed", exit_code=value.get("exit_code"),
                              stdout=value.get("stdout", "")[:4000], stderr=value.get("stderr", "")[:4000],
                              timed_out=bool(value.get("timed_out")), cancelled=bool(value.get("cancelled")),
                              error=value.get("error"))
            except Exception as exc:
                record.update(status="failed", error=str(exc)[:2000])
        results.append(record)
    return results
