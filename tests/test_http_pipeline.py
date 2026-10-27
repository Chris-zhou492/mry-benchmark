import tempfile
import unittest
from pathlib import Path

from memory_benchmark.artifacts import RetrievalArtifact, RetrievalHit, RetrievalRequest, Timing, Usage
from memory_benchmark.evidence import SyntheticWhitespaceCounter, assemble_evidence
from memory_benchmark.memory_http import MemoryHttpClient, MemoryHttpError, RetryPolicy, add_public_history
from memory_benchmark.pipeline import run_memory_stages
from memory_benchmark.state import PROJECT_ROOT, RunConflict, RunStore
from tests.support.mock_memory_server import MockMemoryServer
from tests.support.sample_artifacts import make_sample


class HttpPipelineTests(unittest.TestCase):
    def setUp(self):
        self.server = MockMemoryServer()
        self.client = MemoryHttpClient(self.server.base_url, retry_policy=RetryPolicy(max_attempts=1))
        self.history, self.question, _ = make_sample()
        self.outputs = PROJECT_ROOT / "outputs/test-temporary"
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.outputs)
        self.root = Path(self.temporary.name)
        self.settings = {
            "protocol_version": "0.2.0-draft",
            "method_id": "mock-memory",
            "endpoint_id": "local-mock-v1",
            "evidence_budget": 100,
            "tokenizer": "synthetic-whitespace/0.1.0",
            "retry_max_attempts": 1,
        }

    def tearDown(self):
        self.server.close()
        self.temporary.cleanup()

    def store(self, run_id="http-synthetic", *, resume=False):
        return RunStore(self.root, run_id, self.settings, resume=resume)

    def execute(self, run_id="http-synthetic", *, resume=False, retry_failed=False):
        return run_memory_stages(
            self.store(run_id, resume=resume),
            self.client,
            self.history,
            self.question,
            method_id="mock-memory",
            top_k=10,
            evidence_budget=100,
            tokenizer=SyntheticWhitespaceCounter(),
            retry_failed=retry_failed,
        )

    def test_actual_add_search_requests_are_prefix_safe_and_multi_evidence(self):
        result = self.execute()
        posts = self.server.state.captured
        add = [item["body"] for item in posts if item["path"] == "/add"]
        search = [item["body"] for item in posts if item["path"] == "/search"]
        self.assertEqual(len(add), 1)
        self.assertEqual(len(search), 1)
        self.assertEqual([item["role"] for item in add[0]["messages"]], ["user", "user", "user"])
        self.assertIn("speaker_id:Alice", add[0]["messages"][0]["content"])
        self.assertIn("speaker_id:Bob", add[0]["messages"][1]["content"])
        self.assertIn("timestamp", add[0]["messages"][0])
        self.assertNotIn("timestamp", add[0]["messages"][1])
        self.assertNotIn("timestamp", add[0]["messages"][2])
        serialized = str(posts)
        self.assertNotIn("HIDDEN_PERSONA_DO_NOT_EXPORT", serialized)
        self.assertNotIn("FUTURE_UPDATE", serialized)
        self.assertEqual(len(result.retrieval.hits), 3)
        returned_ids = [item.hit.hit_id for item in result.evidence.presented_hits]
        self.assertEqual(returned_ids, [item.hit_id for item in result.retrieval.hits])
        contents = [item.hit.content for item in result.evidence.presented_hits]
        self.assertTrue(any("jasmine" in item for item in contents))
        self.assertTrue(any("green tea" in item for item in contents))
        self.assertTrue(any("oolong" in item for item in contents))
        self.assertLessEqual(result.evidence.used_tokens, result.evidence.budget_tokens)

    def test_runs_use_isolated_namespaces(self):
        first = self.execute("namespace-run-a")
        second = self.execute("namespace-run-b")
        self.assertNotEqual(first.namespace, second.namespace)
        self.assertEqual(len(first.retrieval.hits), 3)
        self.assertEqual(len(second.retrieval.hits), 3)
        self.assertEqual(self.server.state.request_counts[("POST", "/add")], 2)

    def test_full_run_resume_makes_zero_http_requests(self):
        first = self.execute()
        before = dict(self.server.state.request_counts)
        resumed = self.execute(resume=True)
        self.assertEqual(first.retrieval, resumed.retrieval)
        self.assertEqual(first.evidence, resumed.evidence)
        self.assertEqual(dict(self.server.state.request_counts), before)

    def test_add_retry_reuses_identical_idempotent_payload(self):
        self.server.state.fail_next("/add", 503)
        retry_client = MemoryHttpClient(
            self.server.base_url, retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0), sleeper=lambda _: None
        )
        namespace = "mb-retry-test"
        receipts = add_public_history(
            retry_client,
            self.history,
            self.question,
            run_id="retry-test",
            method_id="mock-memory",
            namespace=namespace,
        )
        add_bodies = [item["body"] for item in self.server.state.captured if item["path"] == "/add"]
        self.assertEqual(len(add_bodies), 2)
        self.assertEqual(add_bodies[0], add_bodies[1])
        self.assertEqual(receipts[0].attempts, 2)
        self.assertEqual(len(self.server.state.messages[namespace]), 3)

    def test_partial_resume_checks_backend_and_retries_failed_search(self):
        self.server.state.fail_next("/search", 503)
        with self.assertRaises(MemoryHttpError):
            self.execute("partial-run")
        resumed = self.execute("partial-run", resume=True, retry_failed=True)
        self.assertEqual(len(resumed.retrieval.hits), 3)
        self.assertEqual(self.server.state.request_counts[("GET", "/state")], 1)
        self.assertEqual(self.server.state.request_counts[("POST", "/add")], 1)
        self.assertEqual(self.server.state.request_counts[("POST", "/search")], 2)

    def test_partial_resume_refuses_lost_backend_state(self):
        self.server.state.fail_next("/search", 503)
        with self.assertRaises(MemoryHttpError):
            self.execute("lost-backend")
        namespace = next(iter(self.server.state.messages))
        self.server.clear_namespace(namespace)
        before_searches = self.server.state.request_counts[("POST", "/search")]
        with self.assertRaises(RunConflict):
            self.execute("lost-backend", resume=True, retry_failed=True)
        self.assertEqual(self.server.state.request_counts[("POST", "/search")], before_searches)

    def test_evidence_budget_skips_oversize_hit_and_keeps_later_rank(self):
        retrieval = RetrievalArtifact(
            schema_version="0.2.0-draft",
            retrieval_id="ret-budget",
            run_id="budget-run",
            question_id="q-budget",
            method_id="mock-memory",
            namespace="mb-budget",
            request=RetrievalRequest("mb-budget", "test", 3, {}),
            hits=(
                RetrievalHit("h1", 1, "too many words in this first hit", 0.9, (), {}),
                RetrievalHit("h2", 2, "short evidence", 0.8, (), {}),
                RetrievalHit("h3", 3, "another short", 0.7, (), {}),
            ),
            status="ok",
            attempts=1,
            usage=Usage(None, None, None, None),
            timing=Timing("2026-10-22T00:00:00Z", 0.0),
        )
        bundle = assemble_evidence(retrieval, budget_tokens=5, tokenizer=SyntheticWhitespaceCounter())
        self.assertEqual([item.hit.hit_id for item in bundle.presented_hits], ["h2"])
        self.assertEqual([item.hit_id for item in bundle.omitted_hits], ["h1", "h3"])
        self.assertEqual(bundle.presented_hits[0].evidence_id, "E1")

    def test_empty_search_is_a_valid_empty_retrieval(self):
        retrieval = self.client.search(
            run_id="empty-run",
            question_id="q-empty",
            method_id="mock-memory",
            namespace="empty-namespace",
            query="no-matching-terms-xyz",
            top_k=5,
        )
        self.assertEqual(retrieval.status, "ok")
        self.assertEqual(retrieval.hits, ())
