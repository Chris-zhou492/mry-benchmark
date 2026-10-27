"""Durable checkpoints and validation shared by the benchmark entry point."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rows(path: Path, *, repair_tail: bool = False) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    offset = 0
    with path.open("rb") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL rows must be objects")
            except (ValueError, UnicodeDecodeError):
                if not repair_tail or line.endswith(b"\n") or stream.read(1):
                    raise ValueError(f"Corrupt JSONL: {path.name}") from None
                # Only a torn final append can be discarded. Complete bad rows fail.
                with path.open("r+b") as output:
                    output.truncate(offset)
                    output.flush()
                    os.fsync(output.fileno())
                break
            rows.append(row)
            offset += len(line)
    if repair_tail and rows and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        # A complete JSON object without its final newline is already committed.
        with path.open("ab") as output:
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
    return rows


@contextmanager
def run_lock(directory: Path):
    """OS locks are released on process exit, including a forced termination."""
    with (directory / ".run.lock").open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("This run is already active in another process") from None
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def row_id(row: dict) -> str:
    value = row.get("id", row.get("idx", row.get("question_id")))
    if value is None or not str(value):
        raise ValueError("Missing question ID")
    return str(value)


def index_rows(path: Path, expected: set[str], *, repair_tail: bool = False) -> dict[str, dict]:
    rows = read_rows(path, repair_tail=repair_tail)
    indexed = {}
    for row in rows:
        ident = row_id(row)
        if ident not in expected or ident in indexed:
            raise ValueError(f"Unexpected or duplicate question ID in {path.name}")
        indexed[ident] = row
    return indexed


def validate_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("API base URL must be http(s), without credentials, query parameters or fragments")
    return value.rstrip("/")


def validate_hits(response: dict, top_k: int) -> list[dict]:
    hits = response.get("data") if isinstance(response, dict) else None
    if not isinstance(hits, list) or len(hits) > top_k:
        raise ValueError("Search response must contain a data list with at most top_k hits")
    if any(
        not isinstance(hit, dict) or not isinstance(hit.get("content"), str) or not hit["content"].strip()
        for hit in hits
    ):
        raise ValueError("Search hits must contain non-empty text content")
    return hits


def judge_failed(row: dict) -> bool:
    reason = str(row.get("rubric_clbench_rationale", ""))
    return reason.startswith(("API call failed", "JSON parse failed", "Unknown error"))


def safe_failure(error: BaseException) -> str:
    # Never persist arbitrary provider response bodies or exception text.
    import subprocess
    from urllib.error import HTTPError, URLError

    current: BaseException | None = error
    while current is not None:
        if isinstance(current, HTTPError):
            return f"HTTP {current.code}: check credentials, endpoint, quota and server logs"
        if isinstance(current, (TimeoutError, URLError)):
            return "Network failure or timeout; check endpoint connectivity and server logs"
        current = current.__cause__
    if isinstance(error, subprocess.CalledProcessError):
        return f"Pipeline exited with code {error.returncode}; inspect pipeline logs"
    if isinstance(error, ValueError):
        if str(error) in {
            "Search response must contain a data list with at most top_k hits",
            "Search hits must contain non-empty text content",
            "Memory returned records for a fresh probe user; check user isolation",
            "ANSWER endpoint returned an invalid chat completion",
            "JUDGE endpoint returned an invalid chat completion",
            "Invalid Add response",
            "Pipeline returned an unexpected question ID/count",
            "Pipeline returned no generated answer",
            "Pipeline returned an invalid score",
        }:
            return str(error)
        return "Invalid data, checkpoint or API response; inspect the stage files and API contract"
    return f"{type(error).__name__}; inspect service/pipeline logs"
