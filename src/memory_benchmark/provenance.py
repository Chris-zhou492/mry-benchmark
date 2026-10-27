"""Stable project-snapshot identities for manifests when Git metadata is absent."""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import ARTIFACT_SCHEMA_VERSION, RunManifest
from .schemas import digest, require
from .state import PROJECT_ROOT

_INCLUDED_ROOTS = ("src", "tests", "tools", "protocols")
_INCLUDED_FILES = ("AGENTS.md", "pyproject.toml")
_EXCLUDED_PARTS = {"__pycache__", ".ruff_cache", ".hatch", ".venv", "outputs"}


def source_file_hashes(root: Path = PROJECT_ROOT) -> dict[str, str]:
    paths = [root / name for name in _INCLUDED_FILES if (root / name).is_file()]
    for directory_name in _INCLUDED_ROOTS:
        directory = root / directory_name
        if directory.is_dir():
            paths.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and not (_EXCLUDED_PARTS & set(path.parts)) and path.suffix not in (".pyc", ".pyo")
            )
    protocol_file = root / "paper" / "pipeline_spec.md"
    if protocol_file.is_file():
        paths.append(protocol_file)
    hashes = {}
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def code_revision(root: Path = PROJECT_ROOT, files: dict[str, str] | None = None) -> str:
    file_hashes = source_file_hashes(root) if files is None else files
    require(bool(file_hashes), "Cannot create a code revision from an empty source snapshot")
    return "snapshot-sha256:" + digest(file_hashes)


def new_validation_manifest(
    *,
    run_id: str,
    question_ids: tuple[str, ...],
    method_id: str,
    endpoint_id: str,
    evidence_budget: int,
    tokenizer_id: str,
    tokenizer_revision: str,
    retry_max_attempts: int,
) -> RunManifest:
    files = source_file_hashes()
    require(bool(files) and "paper/pipeline_spec.md" in files, "Manifest source snapshot is incomplete")
    return RunManifest(
        run_id=run_id,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        protocol_version="0.2.0-draft",
        protocol_sha256=files["paper/pipeline_spec.md"],
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        code_revision=code_revision(files=files),
        source_file_hashes=files,
        loader_versions={},
        split_sha256=digest(list(question_ids)),
        selected_question_ids=question_ids,
        method={
            "method_id": method_id,
            "method_version": "local-mock/0.1.0",
            "endpoint_id": endpoint_id,
            "configuration_sha256": digest({"method_id": method_id, "endpoint_id": endpoint_id}),
            "options": {"execution_scope": "synthetic-only"},
        },
        reader={
            "reader_id": "not-executed",
            "reader_version": "none",
            "configuration_sha256": digest({"status": "not-executed"}),
            "status": "not-executed",
            "reason": "This acceptance run makes no model calls.",
        },
        scorers=(),
        evidence_policy={
            "strategy": "whole_hit",
            "budget_tokens": evidence_budget,
            "tokenizer_id": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
        },
        retry_policy={"max_attempts": retry_max_attempts, "backoff_seconds": 0.0},
        cache_policy={
            "backend_state_verification": "state_endpoint",
            "raw_responses": "not-applicable",
            "model_cache": False,
        },
        environment={"python": platform.python_version(), "implementation": platform.python_implementation()},
        status="validated",
    )
