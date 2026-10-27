"""Run a local-only synthetic Add/Search/evidence acceptance and save its record."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(PROJECT_ROOT))
    from memory_benchmark.evidence import SyntheticWhitespaceCounter
    from memory_benchmark.memory_http import MemoryHttpClient, RetryPolicy
    from memory_benchmark.pipeline import run_memory_stages
    from memory_benchmark.provenance import new_validation_manifest
    from memory_benchmark.state import RunStore, atomic_write
    from tests.support.mock_memory_server import MockMemoryServer
    from tests.support.sample_artifacts import make_sample

    history, question, _ = make_sample()
    run_id = f"mock-http-{uuid.uuid4().hex[:10]}"
    settings = {
        "protocol_version": "0.2.0-draft",
        "method_id": "mock-memory",
        "endpoint_id": "local-mock-v1",
        "evidence_budget": 100,
        "tokenizer": "synthetic-whitespace/0.1.0",
        "retry_max_attempts": 2,
    }
    tokenizer = SyntheticWhitespaceCounter()
    server = MockMemoryServer()
    try:
        client = MemoryHttpClient(server.base_url, retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0))
        first = run_memory_stages(
            RunStore(PROJECT_ROOT / "outputs", run_id, settings),
            client,
            history,
            question,
            method_id="mock-memory",
            top_k=10,
            evidence_budget=100,
            tokenizer=tokenizer,
        )
        calls_before_resume = sum(server.state.request_counts.values())
        resumed = run_memory_stages(
            RunStore(PROJECT_ROOT / "outputs", run_id, settings, resume=True),
            client,
            history,
            question,
            method_id="mock-memory",
            top_k=10,
            evidence_budget=100,
            tokenizer=tokenizer,
        )
        calls_after_resume = sum(server.state.request_counts.values())
        captured = [
            {
                "method": item["method"],
                "path": item["path"],
                "request_body_sha256": hashlib.sha256(item["raw"].encode("utf-8")).hexdigest(),
            }
            for item in server.state.captured
        ]
        serialized_requests = json.dumps(server.state.captured, ensure_ascii=False)
        if "HIDDEN_PERSONA_DO_NOT_EXPORT" in serialized_requests or "FUTURE_UPDATE" in serialized_requests:
            raise RuntimeError("Privacy invariant failed: private or future text reached the mock endpoint")
        if first.retrieval != resumed.retrieval or first.evidence != resumed.evidence:
            raise RuntimeError("Cached resume changed the recorded synthetic artifacts")
        if calls_after_resume != calls_before_resume:
            raise RuntimeError("Complete resume issued unexpected HTTP requests")

        manifest = new_validation_manifest(
            run_id=run_id,
            question_ids=(question.question_id,),
            method_id="mock-memory",
            endpoint_id="local-mock-v1",
            evidence_budget=100,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_revision=tokenizer.tokenizer_revision,
            retry_max_attempts=2,
        )
        output_dir = PROJECT_ROOT / "outputs" / run_id
        atomic_write(output_dir / "run_manifest.json", manifest.to_dict())
        summary = {
            "schema_version": "0.2.0-draft",
            "run_id": run_id,
            "status": "completed-synthetic-http-only",
            "question_count": 1,
            "add_batch_count": len(first.add_receipts),
            "retrieval_hit_count": len(first.retrieval.hits),
            "presented_hit_count": len(first.evidence.presented_hits),
            "omitted_hit_count": len(first.evidence.omitted_hits),
            "evidence_budget_tokens": first.evidence.budget_tokens,
            "evidence_used_tokens": first.evidence.used_tokens,
            "http_request_count": calls_before_resume,
            "request_capture_digests": captured,
            "full_resume_http_request_delta": calls_after_resume - calls_before_resume,
            "real_memory_service_calls": 0,
            "model_calls": 0,
            "real_data_loader_used": False,
            "reader_or_scorer_executed": False,
            "manifest": str(output_dir / "run_manifest.json"),
            "ledger": str(output_dir / "ledger.json"),
        }
        atomic_write(output_dir / "mock_acceptance_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        server.close()


if __name__ == "__main__":
    main()
