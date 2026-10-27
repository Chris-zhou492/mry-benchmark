"""Versioned, strict contracts for the independent evaluation pipeline.

The objects in this module are the full artifact layer.  ``schemas.py`` keeps
the small offline core fixture contract used by the first validation step.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .schemas import Record, SchemaError, canonical, digest, integer, require, sha, text, tuple_of

ARTIFACT_SCHEMA_VERSION = "0.2.0-draft"


def _json(value: Any, name: str) -> None:
    try:
        canonical(value)
    except (TypeError, ValueError) as error:
        raise SchemaError(f"{name} must contain finite JSON values") from error


def _unique(values: tuple[Any, ...], name: str) -> None:
    require(len(values) == len(set(values)), f"Duplicate {name}")


def _fixed_object(value: Any, name: str, required: set[str], optional: set[str] = frozenset()) -> None:
    require(type(value) is dict, f"{name} must be an object")
    require(
        required <= set(value) and set(value) <= required | optional,
        f"{name} fields do not match the manifest contract",
    )


def _optional_text(value: Any, name: str) -> None:
    require(value is None or (type(value) is str and bool(value.strip())), f"{name} must be nonblank text or null")


def _iso8601(value: str, name: str) -> None:
    text(value, name)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise SchemaError(f"{name} must be ISO-8601") from error
    require(parsed.tzinfo is not None, f"{name} must include a timezone")


@dataclass(frozen=True)
class SourceFile(Record):
    path: str
    sha256: str

    def __post_init__(self):
        text(self.path, "source file path")
        sha(self.sha256, "source file sha256")


@dataclass(frozen=True)
class SourceLocator(Record):
    file_sha256: str
    record_pointer: str
    message_pointer: str
    char_start: int | None
    char_end: int | None

    def __post_init__(self):
        sha(self.file_sha256, "file_sha256")
        text(self.record_pointer, "record_pointer")
        text(self.message_pointer, "message_pointer")
        if self.char_start is None:
            require(self.char_end is None, "Character span endpoints must be paired")
        else:
            integer(self.char_start, "char_start")
            integer(self.char_end, "char_end")
            require(self.char_end > self.char_start, "Character span must be nonempty")


@dataclass(frozen=True)
class TimeInfo(Record):
    raw: str | None
    parsed_iso8601: str | None
    timezone: str | None
    precision: str
    parse_status: str

    def __post_init__(self):
        require(self.raw is None or type(self.raw) is str, "Time raw value must be text or null")
        require(self.timezone is None or (type(self.timezone) is str and bool(self.timezone)), "Invalid timezone")
        require(
            self.precision in ("unknown", "year", "month", "day", "minute", "second", "subsecond"),
            "Invalid time precision",
        )
        require(self.parse_status in ("exact", "ambiguous", "missing", "invalid"), "Invalid time parse status")
        require(self.parsed_iso8601 is None or type(self.parsed_iso8601) is str, "Parsed time must be text or null")
        if self.parse_status == "exact":
            require(
                self.raw is not None and self.parsed_iso8601 is not None, "Exact time requires source and parsed values"
            )
            _iso8601(self.parsed_iso8601, "parsed_iso8601")
            require(
                self.timezone is not None and self.precision != "unknown", "Exact time requires timezone and precision"
            )
        elif self.parse_status == "missing":
            require(self.raw is None and self.parsed_iso8601 is None, "Missing time cannot contain a timestamp")
            require(self.timezone is None and self.precision == "unknown", "Missing time metadata must remain unknown")
        else:
            require(self.raw is not None, "Ambiguous or invalid time requires a source value")
            require(self.parsed_iso8601 is None, "Ambiguous or invalid time cannot be normalized")


@dataclass(frozen=True)
class SourceMessageArtifact(Record):
    message_id: str
    session_id: str
    source_order: int
    speaker_id: str
    speaker_type: str
    original_role: str
    visibility: str
    content: str
    content_sha256: str
    source_locator: SourceLocator
    time: TimeInfo

    _nested = {"source_locator": SourceLocator, "time": TimeInfo}

    def __post_init__(self):
        for name in ("message_id", "session_id", "speaker_id", "original_role", "content"):
            text(getattr(self, name), name)
        integer(self.source_order, "source_order")
        require(self.speaker_type in ("human", "assistant", "tool", "system", "unknown"), "Invalid speaker_type")
        require(
            self.visibility
            in ("public_history", "hidden_persona", "public_task_instruction", "other_system", "unclassified_system"),
            "Invalid message visibility",
        )
        sha(self.content_sha256, "content_sha256")
        import hashlib

        require(
            self.content_sha256 == hashlib.sha256(self.content.encode("utf-8")).hexdigest(), "Content hash mismatch"
        )
        require(
            type(self.source_locator) is SourceLocator and type(self.time) is TimeInfo, "Invalid message provenance"
        )
        require(
            self.speaker_type != "system" or self.visibility != "public_history", "System text needs classification"
        )


@dataclass(frozen=True)
class HistoryArtifact(Record):
    schema_version: str
    history_id: str
    family_id: str
    dataset_id: str
    source_files: tuple[SourceFile, ...]
    messages: tuple[SourceMessageArtifact, ...]
    loader_version: str
    content_sha256: str

    _sequences = {"source_files": SourceFile, "messages": SourceMessageArtifact}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported History schema")
        for name in ("history_id", "family_id", "dataset_id", "loader_version"):
            text(getattr(self, name), name)
        tuple_of(self.source_files, SourceFile, "source_files")
        tuple_of(self.messages, SourceMessageArtifact, "messages")
        require(bool(self.source_files) and bool(self.messages), "History requires sources and messages")
        _unique(tuple(item.path for item in self.source_files), "source file paths")
        _unique(tuple(item.message_id for item in self.messages), "message IDs")
        known_source_hashes = {item.sha256 for item in self.source_files}
        require(
            all(item.source_locator.file_sha256 in known_source_hashes for item in self.messages),
            "Message locator references an unlisted source file",
        )
        orders = tuple(item.source_order for item in self.messages)
        require(orders == tuple(sorted(orders)) and len(set(orders)) == len(orders), "History order must be increasing")
        sha(self.content_sha256, "content_sha256")
        require(
            self.content_sha256 == history_content_hash(self.source_files, self.messages),
            "History content hash mismatch",
        )


def history_content_hash(source_files: tuple[SourceFile, ...], messages: tuple[SourceMessageArtifact, ...]) -> str:
    return digest(
        {"source_files": [item.to_dict() for item in source_files], "messages": [item.to_dict() for item in messages]}
    )


@dataclass(frozen=True)
class AllowedHistory(Record):
    last_message_id: str
    last_source_order: int
    prefix_sha256: str

    def __post_init__(self):
        text(self.last_message_id, "last_message_id")
        integer(self.last_source_order, "last_source_order")
        sha(self.prefix_sha256, "prefix_sha256")


@dataclass(frozen=True)
class OptionArtifact(Record):
    option_id: str
    content: str

    def __post_init__(self):
        text(self.option_id, "option_id")
        text(self.content, "option content")


@dataclass(frozen=True)
class PublicQuestionArtifact(Record):
    schema_version: str
    question_id: str
    dataset_id: str
    history_id: str
    family_id: str
    query: str
    task_type: str
    allowed_history: AllowedHistory
    public_instructions: tuple[str, ...]
    options: tuple[OptionArtifact, ...]
    source_locator: SourceLocator

    _nested = {"allowed_history": AllowedHistory, "source_locator": SourceLocator}
    _sequences = {"public_instructions": str, "options": OptionArtifact}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported PublicQuestion schema")
        for name in ("question_id", "dataset_id", "history_id", "family_id", "query"):
            text(getattr(self, name), name)
        require(
            self.task_type in ("choice", "factual_text", "numeric", "rubric_response", "unanswerable"),
            "Invalid task type",
        )
        tuple_of(self.public_instructions, str, "public instructions")
        tuple_of(self.options, OptionArtifact, "options")
        _unique(tuple(item.option_id for item in self.options), "option IDs")
        require(
            len(self.options) >= 2 if self.task_type == "choice" else not self.options, "Options do not match task type"
        )
        require(all(item.strip() for item in self.public_instructions), "Public instructions must be nonblank")


@dataclass(frozen=True)
class Criterion(Record):
    criterion_id: str
    requirement: str
    weight: float
    source: str

    def __post_init__(self):
        text(self.criterion_id, "criterion_id")
        text(self.requirement, "criterion requirement")
        text(self.source, "criterion source")
        require(
            type(self.weight) in (int, float) and math.isfinite(self.weight) and 0 < self.weight,
            "Criterion weight must be positive and finite",
        )


@dataclass(frozen=True)
class PrivateTargetArtifact(Record):
    schema_version: str
    question_id: str
    target_type: str
    answer: Any
    acceptable_answers: tuple[Any, ...]
    criteria: tuple[Criterion, ...]
    numeric_policy: dict[str, Any]
    answerability: str
    origin: SourceLocator

    _nested = {"origin": SourceLocator}
    _sequences = {"criteria": Criterion}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported PrivateTarget schema")
        text(self.question_id, "question_id")
        require(
            self.target_type in ("choice", "factual_text", "numeric", "rubric_response", "unanswerable"),
            "Invalid target_type",
        )
        _json(self.answer, "answer")
        require(type(self.acceptable_answers) is tuple, "acceptable_answers must be immutable")
        for value in self.acceptable_answers:
            _json(value, "acceptable answer")
        tuple_of(self.criteria, Criterion, "criteria")
        _unique(tuple(item.criterion_id for item in self.criteria), "criterion IDs")
        require(type(self.numeric_policy) is dict, "numeric_policy must be an object")
        _json(self.numeric_policy, "numeric_policy")
        require(self.answerability in ("answerable", "unanswerable", "uncertain"), "Invalid answerability")
        require(type(self.origin) is SourceLocator, "Invalid target origin")
        if self.target_type == "choice":
            require(
                type(self.answer) is dict
                and set(self.answer) == {"correct_option_id"}
                and type(self.answer["correct_option_id"]) is str
                and bool(self.answer["correct_option_id"].strip()),
                "Choice target answer must contain a stable correct_option_id",
            )
        if self.target_type == "rubric_response":
            require(bool(self.criteria), "Rubric response target requires fixed criteria")

    @classmethod
    def from_dict(cls, value):
        if type(value) is dict and type(value.get("acceptable_answers")) is list:
            value = dict(value)
            value["acceptable_answers"] = tuple(value["acceptable_answers"])
        return super().from_dict(value)


@dataclass(frozen=True)
class Usage(Record):
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None

    def __post_init__(self):
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None:
                integer(value, name)
        if self.input_tokens is not None and self.output_tokens is not None and self.total_tokens is not None:
            require(self.total_tokens == self.input_tokens + self.output_tokens, "Inconsistent total token count")
        require(
            self.estimated_cost_usd is None
            or (
                type(self.estimated_cost_usd) in (int, float)
                and math.isfinite(self.estimated_cost_usd)
                and self.estimated_cost_usd >= 0
            ),
            "Invalid estimated cost",
        )


@dataclass(frozen=True)
class Timing(Record):
    started_at: str
    duration_ms: float

    def __post_init__(self):
        _iso8601(self.started_at, "started_at")
        require(
            type(self.duration_ms) in (int, float) and math.isfinite(self.duration_ms) and self.duration_ms >= 0,
            "Invalid duration",
        )


@dataclass(frozen=True)
class AddReceipt(Record):
    schema_version: str
    run_id: str
    method_id: str
    namespace: str
    history_prefix_sha256: str
    batch_id: str
    payload_sha256: str
    acknowledged: bool
    backend_state_ref: str | None
    attempts: int
    usage: Usage
    timing: Timing

    _nested = {"usage": Usage, "timing": Timing}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported AddReceipt schema")
        for name in ("run_id", "method_id", "namespace", "batch_id"):
            text(getattr(self, name), name)
        sha(self.history_prefix_sha256, "history_prefix_sha256")
        sha(self.payload_sha256, "payload_sha256")
        require(type(self.acknowledged) is bool and self.acknowledged, "AddReceipt requires explicit acknowledgement")
        require(
            self.backend_state_ref is None or type(self.backend_state_ref) is str, "Invalid backend state reference"
        )
        integer(self.attempts, "attempts")
        require(self.attempts >= 1, "attempts must be positive")
        require(type(self.usage) is Usage and type(self.timing) is Timing, "Invalid AddReceipt metrics")


@dataclass(frozen=True)
class RetrievalRequest(Record):
    user_id: str
    query: str
    top_k: int
    options: dict[str, Any]

    def __post_init__(self):
        text(self.user_id, "user_id")
        text(self.query, "query")
        integer(self.top_k, "top_k")
        require(self.top_k > 0, "top_k must be positive")
        require(type(self.options) is dict, "options must be an object")
        _json(self.options, "options")


@dataclass(frozen=True)
class OriginLink(Record):
    message_id: str
    char_start: int | None
    char_end: int | None

    def __post_init__(self):
        text(self.message_id, "origin message_id")
        if self.char_start is None:
            require(self.char_end is None, "Origin span endpoints must be paired")
        else:
            integer(self.char_start, "origin char_start")
            integer(self.char_end, "origin char_end")
            require(self.char_end > self.char_start, "Origin span must be nonempty")


@dataclass(frozen=True)
class RetrievalHit(Record):
    hit_id: str
    rank: int
    content: str
    method_score: float | None
    origin_links: tuple[OriginLink, ...]
    extensions: dict[str, Any]

    _sequences = {"origin_links": OriginLink}

    def __post_init__(self):
        text(self.hit_id, "hit_id")
        integer(self.rank, "rank")
        require(self.rank >= 1, "hit rank starts at 1")
        text(self.content, "hit content")
        require(
            self.method_score is None or (type(self.method_score) in (int, float) and math.isfinite(self.method_score)),
            "Invalid method_score",
        )
        tuple_of(self.origin_links, OriginLink, "origin_links")
        require(type(self.extensions) is dict, "extensions must be an object")
        _json(self.extensions, "extensions")


@dataclass(frozen=True)
class RetrievalArtifact(Record):
    schema_version: str
    retrieval_id: str
    run_id: str
    question_id: str
    method_id: str
    namespace: str
    request: RetrievalRequest
    hits: tuple[RetrievalHit, ...]
    status: str
    attempts: int
    usage: Usage
    timing: Timing

    _nested = {"request": RetrievalRequest, "usage": Usage, "timing": Timing}
    _sequences = {"hits": RetrievalHit}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported Retrieval schema")
        for name in ("retrieval_id", "run_id", "question_id", "method_id", "namespace"):
            text(getattr(self, name), name)
        tuple_of(self.hits, RetrievalHit, "hits")
        require(
            self.status in ("ok", "client_error", "server_error", "timeout", "invalid_response"),
            "Invalid retrieval status",
        )
        require(self.status == "ok" or not self.hits, "Failed retrieval cannot contain accepted hits")
        require(
            tuple(hit.rank for hit in self.hits) == tuple(range(1, len(self.hits) + 1)), "Hit ranks must be contiguous"
        )
        _unique(tuple(hit.hit_id for hit in self.hits), "hit IDs")
        integer(self.attempts, "attempts")
        require(self.attempts >= 1, "attempts must be positive")


@dataclass(frozen=True)
class PresentedHit(Record):
    evidence_id: str
    hit: RetrievalHit
    token_count: int

    _nested = {"hit": RetrievalHit}

    def __post_init__(self):
        require(re.fullmatch(r"E[1-9][0-9]*", self.evidence_id) is not None, "Invalid evidence_id")
        integer(self.token_count, "token_count")


@dataclass(frozen=True)
class OmittedHit(Record):
    hit_id: str
    rank: int
    reason: str
    token_count: int

    def __post_init__(self):
        text(self.hit_id, "hit_id")
        integer(self.rank, "rank")
        require(self.rank >= 1, "Omitted hit rank starts at 1")
        require(self.reason in ("budget_exceeded", "invalid_content", "policy_excluded"), "Invalid omission reason")
        integer(self.token_count, "token_count")


@dataclass(frozen=True)
class Transformation(Record):
    kind: str
    hit_id: str
    source_start: int | None
    source_end: int | None
    detail: str

    def __post_init__(self):
        text(self.kind, "transformation kind")
        text(self.hit_id, "transformation hit_id")
        text(self.detail, "transformation detail")
        if self.source_start is None:
            require(self.source_end is None, "Transformation offsets must be paired")
        else:
            integer(self.source_start, "source_start")
            integer(self.source_end, "source_end")
            require(self.source_end > self.source_start, "Transformation span must be nonempty")


@dataclass(frozen=True)
class EvidenceBundle(Record):
    schema_version: str
    bundle_id: str
    retrieval_id: str
    question_id: str
    tokenizer_id: str
    tokenizer_revision: str
    budget_tokens: int
    used_tokens: int
    presented_hits: tuple[PresentedHit, ...]
    omitted_hits: tuple[OmittedHit, ...]
    transformations: tuple[Transformation, ...]
    content_sha256: str

    _sequences = {"presented_hits": PresentedHit, "omitted_hits": OmittedHit, "transformations": Transformation}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported EvidenceBundle schema")
        for name in ("bundle_id", "retrieval_id", "question_id", "tokenizer_id", "tokenizer_revision"):
            text(getattr(self, name), name)
        integer(self.budget_tokens, "budget_tokens")
        integer(self.used_tokens, "used_tokens")
        require(self.budget_tokens > 0 and self.used_tokens <= self.budget_tokens, "Evidence budget exceeded")
        tuple_of(self.presented_hits, PresentedHit, "presented_hits")
        tuple_of(self.omitted_hits, OmittedHit, "omitted_hits")
        tuple_of(self.transformations, Transformation, "transformations")
        require(
            self.used_tokens == sum(item.token_count for item in self.presented_hits), "Evidence token total mismatch"
        )
        require(
            tuple(item.evidence_id for item in self.presented_hits)
            == tuple(f"E{index}" for index in range(1, len(self.presented_hits) + 1)),
            "Evidence IDs must be sequential",
        )
        _unique(tuple(item.evidence_id for item in self.presented_hits), "evidence IDs")
        _unique(
            tuple(item.hit.hit_id for item in self.presented_hits) + tuple(item.hit_id for item in self.omitted_hits),
            "bundle hit IDs",
        )
        require(
            tuple(item.hit.rank for item in self.presented_hits)
            == tuple(sorted(item.hit.rank for item in self.presented_hits)),
            "Presented evidence must preserve retrieval order",
        )
        require(
            tuple(item.rank for item in self.omitted_hits) == tuple(sorted(item.rank for item in self.omitted_hits)),
            "Omitted-hit records must preserve retrieval order",
        )
        ranks = tuple(item.hit.rank for item in self.presented_hits) + tuple(item.rank for item in self.omitted_hits)
        require(
            tuple(sorted(ranks)) == tuple(range(1, len(ranks) + 1)), "Evidence bundle must account for retrieval ranks"
        )
        sha(self.content_sha256, "content_sha256")
        require(self.content_sha256 == evidence_content_hash(self), "Evidence bundle hash mismatch")


def evidence_content_hash(bundle: EvidenceBundle) -> str:
    return digest(
        {
            "retrieval_id": bundle.retrieval_id,
            "question_id": bundle.question_id,
            "tokenizer_id": bundle.tokenizer_id,
            "tokenizer_revision": bundle.tokenizer_revision,
            "budget_tokens": bundle.budget_tokens,
            "used_tokens": bundle.used_tokens,
            "presented_hits": [item.to_dict() for item in bundle.presented_hits],
            "omitted_hits": [item.to_dict() for item in bundle.omitted_hits],
            "transformations": [item.to_dict() for item in bundle.transformations],
        }
    )


@dataclass(frozen=True)
class PredictionArtifact(Record):
    schema_version: str
    prediction_id: str
    question_id: str
    bundle_id: str
    reader_config_hash: str
    raw_response_ref: str | None
    answer: str
    selected_option_id: str | None
    abstained: bool
    citations: tuple[str, ...]
    status: str
    attempts: int
    usage: Usage
    timing: Timing

    _nested = {"usage": Usage, "timing": Timing}
    _sequences = {"citations": str}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported Prediction schema")
        for name in ("prediction_id", "question_id", "bundle_id"):
            text(getattr(self, name), name)
        sha(self.reader_config_hash, "reader_config_hash")
        require(self.raw_response_ref is None or type(self.raw_response_ref) is str, "Invalid raw_response_ref")
        require(type(self.answer) is str, "answer must be text")
        require(self.selected_option_id is None or type(self.selected_option_id) is str, "Invalid selected_option_id")
        require(type(self.abstained) is bool, "abstained must be boolean")
        require(not self.abstained or self.selected_option_id is None, "Abstention cannot select an option")
        tuple_of(self.citations, str, "citations")
        require(all(re.fullmatch(r"E[1-9][0-9]*", item) for item in self.citations), "Invalid citation ID")
        _unique(self.citations, "citations")
        require(self.status in ("ok", "format_error", "model_error", "timeout", "refused"), "Invalid prediction status")
        integer(self.attempts, "attempts")
        require(self.attempts >= 1, "attempts must be positive")
        require(
            self.status != "ok" or (self.raw_response_ref is not None and bool(self.raw_response_ref)),
            "Successful prediction needs a raw response reference",
        )


@dataclass(frozen=True)
class CriterionResult(Record):
    criterion_id: str
    decision: str
    score: float | None
    explanation: str

    def __post_init__(self):
        text(self.criterion_id, "criterion_id")
        require(self.decision in ("met", "unmet", "uncertain"), "Invalid criterion decision")
        require(
            self.score is None
            or (type(self.score) in (int, float) and math.isfinite(self.score) and 0 <= self.score <= 1),
            "Invalid criterion score",
        )
        require(type(self.explanation) is str, "explanation must be text")


@dataclass(frozen=True)
class JudgementArtifact(Record):
    schema_version: str
    judgement_id: str
    prediction_id: str
    scorer_id: str
    scorer_version: str
    target_sha256: str
    score: float | None
    criterion_results: tuple[CriterionResult, ...]
    decision: str
    status: str
    raw_response_ref: str | None
    usage: Usage
    timing: Timing

    _nested = {"usage": Usage, "timing": Timing}
    _sequences = {"criterion_results": CriterionResult}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported Judgement schema")
        for name in ("judgement_id", "prediction_id", "scorer_id", "scorer_version"):
            text(getattr(self, name), name)
        sha(self.target_sha256, "target_sha256")
        require(
            self.score is None
            or (type(self.score) in (int, float) and math.isfinite(self.score) and 0 <= self.score <= 1),
            "Invalid judgement score",
        )
        tuple_of(self.criterion_results, CriterionResult, "criterion_results")
        _unique(tuple(item.criterion_id for item in self.criterion_results), "criterion result IDs")
        require(
            self.decision in ("correct", "partially_correct", "incorrect", "unanswerable", "uncertain"),
            "Invalid judgement decision",
        )
        require(
            self.status in ("ok", "judge_uncertain", "judge_failed", "invalid_prediction"), "Invalid judgement status"
        )
        require(self.status == "ok" or self.score is None, "Invalid judgement status cannot carry a main score")
        require(self.raw_response_ref is None or type(self.raw_response_ref) is str, "Invalid raw response reference")


@dataclass(frozen=True)
class EvidenceSpan(Record):
    message_id: str
    char_start: int
    char_end: int
    quote_sha256: str

    def __post_init__(self):
        text(self.message_id, "span message_id")
        integer(self.char_start, "char_start")
        integer(self.char_end, "char_end")
        require(self.char_end > self.char_start, "Evidence span must be nonempty")
        sha(self.quote_sha256, "quote_sha256")


@dataclass(frozen=True)
class RequiredEvidenceSet(Record):
    evidence_set_id: str
    spans: tuple[EvidenceSpan, ...]

    _sequences = {"spans": EvidenceSpan}

    def __post_init__(self):
        text(self.evidence_set_id, "evidence_set_id")
        tuple_of(self.spans, EvidenceSpan, "spans")
        require(bool(self.spans), "Required evidence set must contain a span")


@dataclass(frozen=True)
class AnnotationArtifact(Record):
    schema_version: str
    annotation_id: str
    question_id: str
    source_revision: str
    required_evidence_sets: tuple[RequiredEvidenceSet, ...]
    claims: tuple[dict[str, Any], ...]
    temporal_scope: dict[str, Any]
    state_relations: tuple[dict[str, Any], ...]
    answerability: str
    annotator_status: str
    adjudication_ref: str | None
    version: str

    _sequences = {"required_evidence_sets": RequiredEvidenceSet}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported Annotation schema")
        for name in ("annotation_id", "question_id", "source_revision", "version"):
            text(getattr(self, name), name)
        tuple_of(self.required_evidence_sets, RequiredEvidenceSet, "required_evidence_sets")
        _unique(tuple(item.evidence_set_id for item in self.required_evidence_sets), "evidence set IDs")
        require(
            type(self.claims) is tuple and all(type(item) is dict for item in self.claims), "claims must be objects"
        )
        require(type(self.temporal_scope) is dict, "temporal_scope must be an object")
        require(
            type(self.state_relations) is tuple and all(type(item) is dict for item in self.state_relations),
            "state_relations must be objects",
        )
        _json([*self.claims, self.temporal_scope, *self.state_relations], "annotation content")
        require(self.answerability in ("answerable", "unanswerable", "uncertain"), "Invalid annotation answerability")
        require(self.annotator_status in ("draft", "reviewed", "adjudicated", "uncertain"), "Invalid annotator_status")
        require(self.adjudication_ref is None or type(self.adjudication_ref) is str, "Invalid adjudication_ref")

    @classmethod
    def from_dict(cls, value):
        if type(value) is dict:
            value = dict(value)
            for name in ("claims", "state_relations"):
                if type(value.get(name)) is list:
                    value[name] = tuple(value[name])
        return super().from_dict(value)


@dataclass(frozen=True)
class DiagnosticCheck(Record):
    check_id: str
    result: str
    basis: str
    evidence_refs: tuple[str, ...]

    _sequences = {"evidence_refs": str}

    def __post_init__(self):
        text(self.check_id, "check_id")
        require(self.result in ("met", "unmet", "uncertain", "not_evaluable"), "Invalid diagnostic result")
        text(self.basis, "diagnostic basis")
        tuple_of(self.evidence_refs, str, "evidence_refs")


@dataclass(frozen=True)
class DiagnosisArtifact(Record):
    schema_version: str
    diagnosis_id: str
    question_id: str
    retrieval_id: str
    prediction_id: str
    annotation_version: str | None
    checks: tuple[DiagnosticCheck, ...]
    suspected_stages: tuple[str, ...]
    confidence_basis: str
    intervention_refs: tuple[str, ...]
    status: str

    _sequences = {"checks": DiagnosticCheck, "suspected_stages": str, "intervention_refs": str}

    def __post_init__(self):
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported Diagnosis schema")
        for name in ("diagnosis_id", "question_id", "retrieval_id", "prediction_id", "confidence_basis"):
            text(getattr(self, name), name)
        require(self.annotation_version is None or type(self.annotation_version) is str, "Invalid annotation version")
        tuple_of(self.checks, DiagnosticCheck, "checks")
        tuple_of(self.suspected_stages, str, "suspected_stages")
        require(
            all(
                item
                in (
                    "ingest_api_failure",
                    "retrieval_api_failure",
                    "evidence_missing",
                    "evidence_incomplete",
                    "evidence_assembly_loss",
                    "reading_error",
                    "temporal_state_error",
                    "calculation_error",
                    "judge_failure",
                    "unknown",
                )
                for item in self.suspected_stages
            ),
            "Invalid suspected stage",
        )
        _unique(self.suspected_stages, "suspected stages")
        tuple_of(self.intervention_refs, str, "intervention_refs")
        require(self.status in ("ok", "not_evaluable", "diagnostic_failed"), "Invalid diagnosis status")


@dataclass(frozen=True)
class RunManifest(Record):
    run_id: str
    schema_version: str
    protocol_version: str
    protocol_sha256: str
    created_at: str
    code_revision: str
    source_file_hashes: dict[str, str]
    loader_versions: dict[str, str]
    split_sha256: str
    selected_question_ids: tuple[str, ...]
    method: dict[str, Any]
    reader: dict[str, Any]
    scorers: tuple[dict[str, Any], ...]
    evidence_policy: dict[str, Any]
    retry_policy: dict[str, Any]
    cache_policy: dict[str, Any]
    environment: dict[str, Any]
    status: str

    _sequences = {"selected_question_ids": str}

    def __post_init__(self):
        text(self.run_id, "run_id")
        require(self.schema_version == ARTIFACT_SCHEMA_VERSION, "Unsupported RunManifest schema")
        text(self.protocol_version, "protocol_version")
        sha(self.protocol_sha256, "protocol_sha256")
        _iso8601(self.created_at, "created_at")
        text(self.code_revision, "code_revision")
        require(
            re.fullmatch(r"(?:git:[0-9a-f]{40,64}|snapshot-sha256:[0-9a-f]{64}|unversioned)", self.code_revision)
            is not None,
            "Invalid code_revision",
        )
        for name in (
            "source_file_hashes",
            "loader_versions",
            "method",
            "reader",
            "evidence_policy",
            "retry_policy",
            "cache_policy",
            "environment",
        ):
            value = getattr(self, name)
            require(type(value) is dict, f"{name} must be an object")
            _json(value, name)
        require(bool(self.source_file_hashes), "RunManifest requires source file hashes")
        for path, file_hash in self.source_file_hashes.items():
            text(path, "source file name")
            sha(file_hash, f"source_file_hashes[{path}]")
        require(
            self.source_file_hashes.get("paper/pipeline_spec.md") == self.protocol_sha256,
            "Protocol hash must match the recorded protocol source",
        )
        if self.code_revision.startswith("snapshot-sha256:"):
            require(
                self.code_revision == "snapshot-sha256:" + digest(self.source_file_hashes),
                "Code snapshot revision does not match source hashes",
            )
        for name, version in self.loader_versions.items():
            text(name, "loader name")
            text(version, "loader version")
        tuple_of(self.selected_question_ids, str, "selected_question_ids")
        require(bool(self.selected_question_ids), "RunManifest requires selected questions")
        require(all(item.strip() for item in self.selected_question_ids), "Question IDs must be nonblank")
        _unique(self.selected_question_ids, "selected question IDs")
        require(
            type(self.scorers) is tuple and all(type(item) is dict for item in self.scorers), "scorers must be objects"
        )
        _json(
            [
                self.method,
                self.reader,
                *self.scorers,
                self.evidence_policy,
                self.retry_policy,
                self.cache_policy,
                self.environment,
            ],
            "manifest configuration",
        )
        _fixed_object(
            self.method,
            "method",
            {"method_id", "method_version", "endpoint_id", "configuration_sha256"},
            {"provider", "service_version", "options"},
        )
        for name in ("method_id", "method_version", "endpoint_id"):
            text(self.method[name], f"method.{name}")
        sha(self.method["configuration_sha256"], "method.configuration_sha256")
        for name in ("provider", "service_version"):
            if name in self.method:
                _optional_text(self.method[name], f"method.{name}")
        if "options" in self.method:
            require(type(self.method["options"]) is dict, "method.options must be an object")
        _fixed_object(
            self.reader,
            "reader",
            {"reader_id", "reader_version", "configuration_sha256", "status"},
            {"provider", "requested_model", "actual_model", "prompt_sha256", "model_snapshot", "options", "reason"},
        )
        for name in ("reader_id", "reader_version", "status"):
            text(self.reader[name], f"reader.{name}")
        require(self.reader["status"] in ("configured", "not-executed", "executed", "failed"), "Invalid reader status")
        sha(self.reader["configuration_sha256"], "reader.configuration_sha256")
        for name in ("provider", "requested_model", "actual_model"):
            if name in self.reader:
                _optional_text(self.reader[name], f"reader.{name}")
        if "options" in self.reader:
            require(type(self.reader["options"]) is dict, "reader.options must be an object")
        if "prompt_sha256" in self.reader:
            sha(self.reader["prompt_sha256"], "reader.prompt_sha256")
        for index, scorer in enumerate(self.scorers):
            _fixed_object(
                scorer,
                f"scorers[{index}]",
                {"scorer_id", "scorer_version", "configuration_sha256", "status"},
                {"provider", "requested_model", "actual_model", "prompt_sha256", "options"},
            )
            for name in ("scorer_id", "scorer_version", "status"):
                text(scorer[name], f"scorers[{index}].{name}")
            require(
                scorer["status"] in ("configured", "not-executed", "executed", "failed"),
                f"Invalid scorer status at index {index}",
            )
            sha(scorer["configuration_sha256"], f"scorers[{index}].configuration_sha256")
            for name in ("provider", "requested_model", "actual_model"):
                if name in scorer:
                    _optional_text(scorer[name], f"scorers[{index}].{name}")
            if "options" in scorer:
                require(type(scorer["options"]) is dict, f"scorers[{index}].options must be an object")
            if "prompt_sha256" in scorer:
                sha(scorer["prompt_sha256"], f"scorers[{index}].prompt_sha256")
        _fixed_object(
            self.evidence_policy,
            "evidence_policy",
            {"strategy", "budget_tokens", "tokenizer_id", "tokenizer_revision"},
            {"prompt_reserve_tokens", "output_reserve_tokens", "overflow_policy"},
        )
        require(self.evidence_policy["strategy"] in ("whole_hit", "fixed_slice"), "Invalid evidence strategy")
        integer(self.evidence_policy["budget_tokens"], "evidence_policy.budget_tokens")
        require(self.evidence_policy["budget_tokens"] > 0, "Evidence budget must be positive")
        text(self.evidence_policy["tokenizer_id"], "evidence_policy.tokenizer_id")
        text(self.evidence_policy["tokenizer_revision"], "evidence_policy.tokenizer_revision")
        for name in ("prompt_reserve_tokens", "output_reserve_tokens"):
            if name in self.evidence_policy:
                integer(self.evidence_policy[name], f"evidence_policy.{name}")
        if "overflow_policy" in self.evidence_policy:
            text(self.evidence_policy["overflow_policy"], "evidence_policy.overflow_policy")
        _fixed_object(self.retry_policy, "retry_policy", {"max_attempts", "backoff_seconds"}, {"retry_statuses"})
        integer(self.retry_policy["max_attempts"], "retry_policy.max_attempts")
        require(1 <= self.retry_policy["max_attempts"] <= 10, "Invalid retry attempt limit")
        require(
            type(self.retry_policy["backoff_seconds"]) in (int, float) and self.retry_policy["backoff_seconds"] >= 0,
            "Invalid retry backoff",
        )
        if "retry_statuses" in self.retry_policy:
            statuses = self.retry_policy["retry_statuses"]
            require(
                type(statuses) in (list, tuple) and all(type(code) is int and 100 <= code <= 599 for code in statuses),
                "Invalid retry statuses",
            )
        _fixed_object(
            self.cache_policy,
            "cache_policy",
            {"backend_state_verification", "raw_responses"},
            {"model_cache", "key_version"},
        )
        require(
            self.cache_policy["backend_state_verification"] in ("snapshot", "state_endpoint", "rebuild", "disabled"),
            "Invalid backend state cache policy",
        )
        require(
            self.cache_policy["raw_responses"] in ("retain", "redact", "not-applicable"),
            "Invalid raw response cache policy",
        )
        if "model_cache" in self.cache_policy:
            require(type(self.cache_policy["model_cache"]) is bool, "cache_policy.model_cache must be boolean")
        if "key_version" in self.cache_policy:
            text(self.cache_policy["key_version"], "cache_policy.key_version")
        _fixed_object(
            self.environment,
            "environment",
            {"python", "implementation"},
            {"platform", "machine", "dependency_versions"},
        )
        text(self.environment["python"], "environment.python")
        text(self.environment["implementation"], "environment.implementation")
        for name in ("platform", "machine"):
            if name in self.environment:
                _optional_text(self.environment[name], f"environment.{name}")
        if "dependency_versions" in self.environment:
            versions = self.environment["dependency_versions"]
            require(type(versions) is dict, "environment.dependency_versions must be an object")
            for name, version in versions.items():
                text(name, "dependency name")
                text(version, f"environment.dependency_versions[{name}]")
        require(
            self.status in ("planned", "validated", "running", "partial", "completed", "failed", "aborted"),
            "Invalid run status",
        )
        _reject_secret_keys(
            {
                "method": self.method,
                "reader": self.reader,
                "scorers": self.scorers,
                "evidence_policy": self.evidence_policy,
                "retry_policy": self.retry_policy,
                "cache_policy": self.cache_policy,
                "environment": self.environment,
            }
        )

    @classmethod
    def from_dict(cls, value):
        if type(value) is dict and type(value.get("scorers")) is list:
            value = dict(value)
            value["scorers"] = tuple(value["scorers"])
        return super().from_dict(value)


def _reject_secret_keys(value: Any) -> None:
    forbidden = {"api_key", "authorization", "password", "secret", "access_token", "refresh_token"}
    if isinstance(value, dict):
        for key, item in value.items():
            lower = key.lower()
            if lower in forbidden or lower.endswith(("_api_key", "_password", "_secret", "_token")):
                if item not in (None, ""):
                    raise SchemaError("Credentials must not be stored in a RunManifest")
            _reject_secret_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secret_keys(item)


ROOT_ARTIFACTS = (
    HistoryArtifact,
    PublicQuestionArtifact,
    PrivateTargetArtifact,
    AddReceipt,
    RetrievalArtifact,
    EvidenceBundle,
    PredictionArtifact,
    JudgementArtifact,
    AnnotationArtifact,
    DiagnosisArtifact,
    RunManifest,
)
