import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_benchmark.core import smoke
from memory_benchmark.schemas import SchemaError, digest, loads
from memory_benchmark.state import PROJECT_ROOT, RunConflict, RunLocked, RunStore


class StateTests(unittest.TestCase):
    def setUp(self):
        self.output = PROJECT_ROOT / "outputs/test-temporary"
        self.output.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.output)
        self.root = Path(self.temporary.name)
        self.settings = {"protocol_version": "test/0.1.0", "evidence_budget": 100}

    def tearDown(self):
        self.temporary.cleanup()

    def store(self, *, resume=False, settings=None):
        return RunStore(self.root, "test-run", self.settings if settings is None else settings, resume=resume)

    def test_output_cannot_escape_project(self):
        with self.assertRaises(RunConflict):
            RunStore(PROJECT_ROOT.parent / "outside-artifacts", "test", self.settings)
        with self.assertRaises(SchemaError):
            RunStore(self.root, "../escape", self.settings)

    def test_requires_lock(self):
        with self.assertRaises(RunLocked):
            self.store().begin("x", digest("x"))

    def test_concurrent_writer_refused_and_lock_released(self):
        with self.store().locked():
            with self.assertRaises(RunLocked):
                with self.store(resume=True).locked():
                    pass
        self.assertFalse((self.root / "test-run/run.lock").exists())

    def test_resume_skips_completed_operation(self):
        calls = []

        def operation():
            calls.append(1)
            return {"score": 1.0}

        with self.store().locked() as store:
            self.assertEqual(store.run_step("score", "input", operation), {"score": 1.0})
        with self.store(resume=True).locked() as store:
            self.assertEqual(store.run_step("score", "input", operation), {"score": 1.0})
        self.assertEqual(calls, [1])

    def test_existing_run_requires_resume(self):
        with self.store().locked():
            pass
        with self.assertRaises(RunConflict):
            with self.store().locked():
                pass

    def test_absent_run_cannot_resume(self):
        with self.assertRaises(RunConflict):
            with self.store(resume=True).locked():
                pass

    def test_changed_config_and_stage_input_rejected(self):
        with self.store().locked() as store:
            store.run_step("x", "original", lambda: {"value": 1})
        with self.assertRaises(RunConflict):
            with self.store(resume=True, settings={"protocol_version": "changed"}).locked():
                pass
        with self.store(resume=True).locked() as store:
            with self.assertRaises(RunConflict):
                store.run_step("x", "changed", lambda: {"value": 2})

    def test_failed_stage_not_a_successful_zero(self):
        with self.store().locked() as store:

            def fail():
                raise RuntimeError("SECRET_PROVIDER_MESSAGE")

            with self.assertRaises(RuntimeError):
                store.run_step("judge", "input", fail)
            entry = store.snapshot()["stages"]["judge"]
            self.assertEqual(entry["status"], "failed")
            self.assertIsNone(entry["artifact"])
            self.assertNotIn("SECRET_PROVIDER_MESSAGE", (store.path / "ledger.json").read_text())
            with self.assertRaises(RunConflict):
                store.begin("judge", digest("input"))
            self.assertTrue(store.begin("judge", digest("input"), retry=True))
            store.complete("judge", {"score": None, "status": "judge_failed"})
            self.assertEqual(store.snapshot()["stages"]["judge"]["attempts"], 2)

    def test_interrupted_stage_needs_explicit_recovery(self):
        with self.store().locked() as store:
            store.begin("x", digest("input"))
        with self.store(resume=True).locked() as store:
            with self.assertRaises(RunConflict):
                store.begin("x", digest("input"), retry=True)
            store.recover_interrupted("x", digest("input"))
            self.assertTrue(store.begin("x", digest("input"), retry=True))
            store.complete("x", {"value": 1})

    def test_completed_artifact_tampering_rejected(self):
        with self.store().locked() as store:
            store.run_step("x", "input", lambda: {"score": 1})
            path = store.path / "ledger.json"
        payload = loads(path.read_text())
        payload["stages"]["x"]["artifact"]["score"] = 0
        path.write_text(json.dumps(payload))
        with self.assertRaises(RunConflict):
            with self.store(resume=True).locked():
                pass

    def test_truncated_ledger_fails_closed(self):
        with self.store().locked() as store:
            path = store.path / "ledger.json"
        path.write_text('{"schema_version":')
        with self.assertRaises(SchemaError):
            with self.store(resume=True).locked():
                pass

    def test_disk_commit_failure_cannot_be_reused_as_success(self):
        with self.store().locked() as store:
            store.begin("x", digest("input"))
            with patch("memory_benchmark.state.atomic_write", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    store.complete("x", {"score": 1.0})
            with self.assertRaises(RunLocked):
                store.snapshot()
        with self.store(resume=True).locked() as store:
            entry = store.snapshot()["stages"]["x"]
            self.assertEqual(entry["status"], "running")
            self.assertIsNone(entry["artifact"])
            with self.assertRaises(RunConflict):
                store.begin("x", digest("input"))

    def test_credentials_never_persist(self):
        with self.assertRaises(SchemaError):
            self.store(settings={"reader": {"api_key": "secret"}})
        with self.store().locked() as store:
            store.begin("x", digest("input"))
            with self.assertRaises(SchemaError):
                store.complete("x", {"Authorization": "Bearer secret"})

    def test_smoke_is_offline_and_repeatable(self):
        fixture = PROJECT_ROOT / "tests/fixtures/core_choice.json"
        with (
            patch("socket.create_connection", side_effect=AssertionError("Network forbidden")),
            patch("socket.socket", side_effect=AssertionError("Network forbidden")),
        ):
            result = smoke(fixture, self.root, "synthetic")
            again = smoke(fixture, self.root, "synthetic", resume=True)
        self.assertEqual(result, again)
        self.assertEqual(result["public_message_count"], 3)
        self.assertEqual(result["judgement"]["score"], 1.0)
        self.assertEqual(result["memory_http_calls"], 0)
        ledger = Path(result["ledger"]).read_text()
        self.assertNotIn("HIDDEN_PERSONA_DO_NOT_EXPORT", ledger)
        self.assertNotIn("PRIVATE_TARGET_DO_NOT_EXPORT", ledger)

    def test_core_runtime_does_not_import_legacy_packages(self):
        import ast

        package = PROJECT_ROOT / "src/memory_benchmark"
        forbidden = {"memory_service", "api_config", "run_benchmark", "text_memory_runner", "mem0"}
        for path in package.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    self.assertFalse({item.name.split(".")[0] for item in node.names} & forbidden)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    self.assertNotIn(node.module.split(".")[0], forbidden)
