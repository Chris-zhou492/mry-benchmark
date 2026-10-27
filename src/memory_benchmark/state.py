"""Local stage ledger: explicit locking, immutable inputs and atomic completion."""

import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schemas import VERSION, SchemaError, canonical, digest, loads, require, sha

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RunConflict(RuntimeError):
    pass


class RunLocked(RuntimeError):
    pass


def safe_settings(value):
    """Reject credential-bearing configuration; never scrub silently."""
    canonical(value)
    if isinstance(value, dict):
        for key, item in value.items():
            secret = key.lower() in ("api_key", "authorization", "password", "secret", "access_token")
            secret = secret or key.lower().endswith(("_api_key", "_password", "_secret"))
            if secret and item not in (None, ""):
                raise SchemaError("Credentials must not be persisted")
            safe_settings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            safe_settings(item)


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def scoped_output(value):
    path = Path(value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve() / "outputs"):
        raise RunConflict("Run artifacts must stay inside this project's outputs directory")
    return path


class RunStore:
    """Core ledger, not the full protocol RunManifest or a Memory backend."""

    def __init__(self, output_root, run_id, settings, *, resume=False):
        require(
            type(run_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) is not None,
            "Invalid run ID",
        )
        safe_settings(settings)
        self.settings = loads(canonical(settings))
        self.identity = digest({"schema_version": VERSION, "settings": self.settings})
        self.path = scoped_output(Path(output_root) / run_id)
        self.resume = resume
        self.token = None
        self.state = None

    @contextmanager
    def locked(self):
        if self.token is not None:
            raise RunLocked("Store already holds a lock")
        self.path.mkdir(parents=True, exist_ok=True)
        lock = self.path / "run.lock"
        token = uuid.uuid4().hex
        try:
            stream = lock.open("x", encoding="utf-8")
        except FileExistsError:
            raise RunLocked("Run lock exists; stale locks require explicit operator review") from None
        self.token = token
        try:
            with stream:
                stream.write(canonical({"pid": os.getpid(), "token": token, "created_at": utc_now()}))
                stream.flush()
                os.fsync(stream.fileno())
            ledger = self.path / "ledger.json"
            if ledger.exists():
                if not self.resume:
                    raise RunConflict("Existing run requires resume")
                self.state = loads(ledger.read_text(encoding="utf-8"))
                if (
                    type(self.state) is not dict
                    or self.state.get("identity") != self.identity
                    or self.state.get("run_id") != self.path.name
                ):
                    raise RunConflict("Input/configuration identity changed")
                self._validate()
            else:
                if self.resume:
                    raise RunConflict("Cannot resume an absent ledger")
                self.state = {
                    "schema_version": VERSION,
                    "run_id": self.path.name,
                    "identity": self.identity,
                    "settings": self.settings,
                    "created_at": utc_now(),
                    "stages": {},
                }
                self._save()
            yield self
        finally:
            self.state = None
            self.token = None
            if lock.exists() and loads(lock.read_text(encoding="utf-8")).get("token") == token:
                lock.unlink()

    def _validate(self):
        if type(self.state) is not dict or set(self.state) != {
            "schema_version",
            "run_id",
            "identity",
            "settings",
            "created_at",
            "stages",
        }:
            raise RunConflict("Invalid ledger schema")
        if (
            self.state["schema_version"] != VERSION
            or digest({"schema_version": VERSION, "settings": self.state["settings"]}) != self.identity
            or type(self.state["stages"]) is not dict
        ):
            raise RunConflict("Ledger identity mismatch")
        for entry in self.state["stages"].values():
            if type(entry) is not dict or set(entry) != {
                "input_sha256",
                "status",
                "attempts",
                "artifact",
                "artifact_sha256",
                "error_code",
            }:
                raise RunConflict("Invalid stage schema")
            sha(entry["input_sha256"], "input_sha256")
            if type(entry["attempts"]) is not int or entry["attempts"] < 1:
                raise RunConflict("Invalid attempt counter")
            if entry["status"] not in ("running", "succeeded", "failed"):
                raise RunConflict("Invalid stage status")
            if entry["status"] == "succeeded":
                if digest(entry["artifact"]) != entry["artifact_sha256"] or entry["error_code"] is not None:
                    raise RunConflict("Completed artifact integrity check failed")
            elif entry["artifact"] is not None or entry["artifact_sha256"] is not None:
                raise RunConflict("Unfinished stage cannot have a successful artifact")

    def _active(self):
        if self.token is None or self.state is None:
            raise RunLocked("A run lock is required")

    def _save(self):
        self._active()
        try:
            atomic_write(self.path / "ledger.json", self.state)
        except Exception:
            # Never let an uncommitted in-memory succeeded flag serve a cache hit.
            # Re-open under a new lock and inspect the persisted state to recover.
            self.state = None
            raise

    def begin(self, stage_id, input_sha256, *, retry=False):
        self._active()
        require(type(stage_id) is str and stage_id.strip(), "Invalid stage ID")
        sha(input_sha256, "input_sha256")
        previous = self.state["stages"].get(stage_id)
        if previous:
            if previous["input_sha256"] != input_sha256:
                raise RunConflict("Stage input changed")
            if previous["status"] == "succeeded":
                return False
            if previous["status"] == "running" or not retry:
                raise RunConflict("Unfinished stage requires explicit recovery/retry")
        self.state["stages"][stage_id] = {
            "input_sha256": input_sha256,
            "status": "running",
            "attempts": previous["attempts"] + 1 if previous else 1,
            "artifact": None,
            "artifact_sha256": None,
            "error_code": None,
        }
        self._save()
        return True

    def complete(self, stage_id, artifact):
        self._active()
        entry = self.state["stages"].get(stage_id)
        if not entry or entry["status"] != "running":
            raise RunConflict("Only a running stage may complete")
        safe_settings(artifact)
        result = loads(canonical(artifact))
        entry.update(status="succeeded", artifact=result, artifact_sha256=digest(result), error_code=None)
        # Artifact and succeeded flag commit in one atomic file replacement.
        self._save()

    def fail(self, stage_id, error_code):
        self._active()
        require(
            type(error_code) is str and re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", error_code) is not None,
            "Use a safe error code, not raw provider error text",
        )
        entry = self.state["stages"].get(stage_id)
        if not entry or entry["status"] != "running":
            raise RunConflict("Only a running stage may fail")
        entry.update(status="failed", error_code=error_code)
        self._save()

    def recover_interrupted(self, stage_id, input_sha256):
        self._active()
        entry = self.state["stages"].get(stage_id)
        if not entry or entry["status"] != "running" or entry["input_sha256"] != input_sha256:
            raise RunConflict("Not a matching interrupted stage")
        self.fail(stage_id, "INTERRUPTED")

    def run_step(self, stage_id, input_value, operation, *, retry=False):
        if not self.begin(stage_id, digest(input_value), retry=retry):
            return loads(canonical(self.state["stages"][stage_id]["artifact"]))
        try:
            artifact = operation()
            self.complete(stage_id, artifact)
            return loads(canonical(artifact))
        except Exception:
            if self.state is not None and self.state["stages"][stage_id]["status"] == "running":
                self.fail(stage_id, "OPERATION_FAILED")
            raise

    def snapshot(self):
        self._active()
        return loads(canonical(self.state))
