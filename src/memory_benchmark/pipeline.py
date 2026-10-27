"""Question-level scheduler for Add → Search → evidence assembly.

Answer generation and scoring remain separate downstream stages.  This module
does not invoke a model or make any decision using a private target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .artifacts import AddReceipt, EvidenceBundle, HistoryArtifact, PublicQuestionArtifact, RetrievalArtifact
from .evidence import TokenCounter, assemble_evidence
from .memory_http import MemoryHttpClient, add_public_history, namespace_for, validated_public_prefix
from .state import RunConflict, RunStore


@dataclass(frozen=True)
class MemoryStageResult:
    namespace: str
    add_receipts: tuple[AddReceipt, ...]
    retrieval: RetrievalArtifact
    evidence: EvidenceBundle


def run_memory_stages(
    store: RunStore,
    client: MemoryHttpClient,
    history: HistoryArtifact,
    question: PublicQuestionArtifact,
    *,
    method_id: str,
    top_k: int,
    evidence_budget: int,
    tokenizer: TokenCounter,
    search_options: dict[str, Any] | None = None,
    retry_failed: bool = False,
) -> MemoryStageResult:
    """Run and checkpoint the public Memory and evidence stages for one question."""
    prefix = validated_public_prefix(history, question)
    namespace = namespace_for(store.path.name, method_id, question.question_id, question.allowed_history.prefix_sha256)
    ingest_input = {
        "namespace": namespace,
        "method_id": method_id,
        "history_id": history.history_id,
        "prefix_sha256": question.allowed_history.prefix_sha256,
        "messages": [item.to_dict() for item in prefix],
    }
    with store.locked() as active:
        stages_before = active.snapshot()["stages"]
        ingest_was_cached = stages_before.get("ingest", {}).get("status") == "succeeded"
        receipts_json = active.run_step(
            "ingest",
            ingest_input,
            lambda: [
                item.to_dict()
                for item in add_public_history(
                    client,
                    history,
                    question,
                    run_id=store.path.name,
                    method_id=method_id,
                    namespace=namespace,
                )
            ],
            retry=retry_failed,
        )
        receipts = tuple(AddReceipt.from_dict(item) for item in receipts_json)
        if not receipts:
            raise RunConflict("Ingest stage has no acknowledged Add receipts")

        current = active.snapshot()["stages"]
        retrieval_was_cached = current.get("retrieve", {}).get("status") == "succeeded"
        if ingest_was_cached and not retrieval_was_cached:
            expected_ref = receipts[-1].backend_state_ref
            if expected_ref is None:
                raise RunConflict("Cannot resume a partial run without a verifiable backend state reference")
            actual_ref = client.verify_namespace(namespace)
            if actual_ref != expected_ref:
                raise RunConflict("Backend state changed or was lost; start a new run to rebuild the namespace")

        retrieval_input = {
            "namespace": namespace,
            "method_id": method_id,
            "question_id": question.question_id,
            "query": question.query,
            "top_k": top_k,
            "options": search_options or {},
        }
        retrieval_json = active.run_step(
            "retrieve",
            retrieval_input,
            lambda: client.search(
                run_id=store.path.name,
                question_id=question.question_id,
                method_id=method_id,
                namespace=namespace,
                query=question.query,
                top_k=top_k,
                options=search_options,
            ).to_dict(),
            retry=retry_failed,
        )
        retrieval = RetrievalArtifact.from_dict(retrieval_json)

        evidence_input = {
            "retrieval": retrieval.to_dict(),
            "budget_tokens": evidence_budget,
            "tokenizer_id": tokenizer.tokenizer_id,
            "tokenizer_revision": tokenizer.tokenizer_revision,
        }
        evidence_json = active.run_step(
            "assemble_evidence",
            evidence_input,
            lambda: assemble_evidence(retrieval, budget_tokens=evidence_budget, tokenizer=tokenizer).to_dict(),
            retry=retry_failed,
        )
        evidence = EvidenceBundle.from_dict(evidence_json)
    return MemoryStageResult(namespace, receipts, retrieval, evidence)
