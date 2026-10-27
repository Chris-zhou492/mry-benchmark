"""Evidence-budget assembly that preserves the Memory method's hit order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    EvidenceBundle,
    OmittedHit,
    PresentedHit,
    RetrievalArtifact,
    Transformation,
)
from .schemas import digest, require, text


class TokenCounter(Protocol):
    tokenizer_id: str
    tokenizer_revision: str

    def count(self, text_value: str) -> int: ...


@dataclass(frozen=True)
class SyntheticWhitespaceCounter:
    """Test-only counter; never claim its counts are model tokenizer tokens."""

    tokenizer_id: str = "synthetic-whitespace"
    tokenizer_revision: str = "0.1.0"

    def count(self, text_value: str) -> int:
        return len(text_value.split())


def assemble_evidence(retrieval: RetrievalArtifact, *, budget_tokens: int, tokenizer: TokenCounter) -> EvidenceBundle:
    require(
        type(retrieval) is RetrievalArtifact and retrieval.status == "ok",
        "Evidence assembly needs a successful retrieval",
    )
    require(type(budget_tokens) is int and budget_tokens > 0, "Evidence budget must be positive")
    text(tokenizer.tokenizer_id, "tokenizer_id")
    text(tokenizer.tokenizer_revision, "tokenizer_revision")
    presented: list[PresentedHit] = []
    omitted: list[OmittedHit] = []
    used = 0
    for hit in retrieval.hits:
        evidence_id = f"E{len(presented) + 1}"
        count = tokenizer.count(f"[{evidence_id}]\n{hit.content}")
        require(type(count) is int and count >= 0, "Token counter must return a nonnegative integer")
        if used + count <= budget_tokens:
            presented.append(PresentedHit(evidence_id, hit, count))
            used += count
        else:
            omitted.append(OmittedHit(hit.hit_id, hit.rank, "budget_exceeded", count))

    presented_tuple = tuple(presented)
    omitted_tuple = tuple(omitted)
    transforms: tuple[Transformation, ...] = ()
    hash_input = {
        "retrieval_id": retrieval.retrieval_id,
        "question_id": retrieval.question_id,
        "tokenizer_id": tokenizer.tokenizer_id,
        "tokenizer_revision": tokenizer.tokenizer_revision,
        "budget_tokens": budget_tokens,
        "used_tokens": used,
        "presented_hits": [item.to_dict() for item in presented_tuple],
        "omitted_hits": [item.to_dict() for item in omitted_tuple],
        "transformations": [],
    }
    content_hash = digest(hash_input)
    bundle_id = "bundle-" + digest({"content_sha256": content_hash})[:32]
    return EvidenceBundle(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        bundle_id=bundle_id,
        retrieval_id=retrieval.retrieval_id,
        question_id=retrieval.question_id,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.tokenizer_revision,
        budget_tokens=budget_tokens,
        used_tokens=used,
        presented_hits=presented_tuple,
        omitted_hits=omitted_tuple,
        transformations=transforms,
        content_sha256=content_hash,
    )
