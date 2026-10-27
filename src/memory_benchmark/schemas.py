"""Strict, immutable core contracts; no legacy code or third-party dependencies."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, fields

VERSION = "0.1.0"


class SchemaError(ValueError):
    pass


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError("Non-finite numbers are forbidden")
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise SchemaError("JSON keys must be strings")
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def canonical(value):
    _finite(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(SchemaError("Invalid number")),
        )
        _finite(value)
        return value
    except (TypeError, json.JSONDecodeError) as error:
        raise SchemaError("Invalid JSON") from error


def require(condition, message):
    if not condition:
        raise SchemaError(message)


def text(value, name):
    require(type(value) is str and bool(value.strip()), f"{name} must be nonblank text")


def sha(value, name):
    require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{name} must be a SHA-256 hex string"
    )


def integer(value, name):
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")


def tuple_of(value, kind, name):
    require(type(value) is tuple and all(type(item) is kind for item in value), f"Invalid {name}")


class Record:
    @classmethod
    def from_dict(cls, value):
        require(type(value) is dict, f"{cls.__name__} must be an object")
        expected = {field.name for field in fields(cls)}
        require(set(value) == expected, f"{cls.__name__} fields do not match the schema")
        converted = dict(value)
        for name, kind in getattr(cls, "_nested", {}).items():
            converted[name] = kind.from_dict(value[name])
        for name, kind in getattr(cls, "_sequences", {}).items():
            require(type(value[name]) is list, f"{name} must be an array")
            converted[name] = tuple(kind.from_dict(item) if issubclass(kind, Record) else item for item in value[name])
        return cls(**converted)

    def to_dict(self):
        # JSON round-trip converts immutable tuples back to public JSON arrays.
        return loads(canonical(asdict(self)))


@dataclass(frozen=True)
class SourceLocator(Record):
    file_sha256: str
    record_pointer: str
    message_pointer: str

    def __post_init__(self):
        sha(self.file_sha256, "file_sha256")
        text(self.record_pointer, "record_pointer")
        text(self.message_pointer, "message_pointer")


@dataclass(frozen=True)
class SourceMessage(Record):
    message_id: str
    session_id: str
    source_order: int
    speaker_id: str
    speaker_type: str
    original_role: str
    visibility: str
    content: str
    content_sha256: str
    source_time_raw: str | None
    source_locator: SourceLocator

    _nested = {"source_locator": SourceLocator}

    def __post_init__(self):
        for name in ("message_id", "session_id", "speaker_id", "original_role", "content"):
            text(getattr(self, name), name)
        integer(self.source_order, "source_order")
        require(self.speaker_type in ("human", "assistant", "tool", "system", "unknown"), "Invalid speaker_type")
        require(
            self.visibility in ("public_history", "hidden_persona", "public_task_instruction", "unclassified_system"),
            "Invalid visibility",
        )
        require(self.source_time_raw is None or type(self.source_time_raw) is str, "Invalid source time")
        require(type(self.source_locator) is SourceLocator, "Invalid source locator")
        sha(self.content_sha256, "content_sha256")
        require(self.content_sha256 == text_hash(self.content), "Content hash mismatch")
        require(
            self.speaker_type != "system" or self.visibility != "public_history", "System text needs classification"
        )


@dataclass(frozen=True)
class History(Record):
    history_id: str
    family_id: str
    dataset_id: str
    loader_version: str
    messages: tuple[SourceMessage, ...]

    _sequences = {"messages": SourceMessage}

    def __post_init__(self):
        for name in ("history_id", "family_id", "dataset_id", "loader_version"):
            text(getattr(self, name), name)
        tuple_of(self.messages, SourceMessage, "messages")
        require(bool(self.messages), "Empty history")
        require(len({item.message_id for item in self.messages}) == len(self.messages), "Duplicate message IDs")
        orders = [item.source_order for item in self.messages]
        require(orders == sorted(set(orders)), "History order must be strictly increasing")

    def public_prefix(self, last_message_id):
        ids = [item.message_id for item in self.messages]
        require(last_message_id in ids, "Unknown cutoff message")
        prefix = self.messages[: ids.index(last_message_id) + 1]
        require(not any(item.visibility == "unclassified_system" for item in prefix), "Unclassified system input")
        return tuple(item for item in prefix if item.visibility == "public_history")

    def prefix_hash(self, last_message_id):
        return digest([item.to_dict() for item in self.public_prefix(last_message_id)])


@dataclass(frozen=True)
class Option(Record):
    option_id: str
    content: str

    def __post_init__(self):
        text(self.option_id, "option_id")
        text(self.content, "option content")


@dataclass(frozen=True)
class PublicQuestion(Record):
    question_id: str
    dataset_id: str
    history_id: str
    family_id: str
    query: str
    task_type: str
    last_message_id: str
    prefix_sha256: str
    public_instructions: str
    options: tuple[Option, ...]

    _sequences = {"options": Option}

    def __post_init__(self):
        for name in ("question_id", "dataset_id", "history_id", "family_id", "query", "last_message_id"):
            text(getattr(self, name), name)
        require(type(self.public_instructions) is str, "Instructions must be text")
        require(
            self.task_type in ("choice", "factual_text", "numeric", "rubric_response", "unanswerable"),
            "Invalid task type",
        )
        sha(self.prefix_sha256, "prefix_sha256")
        tuple_of(self.options, Option, "options")
        require(len({item.option_id for item in self.options}) == len(self.options), "Duplicate option IDs")
        require(
            len(self.options) >= 2 if self.task_type == "choice" else not self.options, "Options do not match task type"
        )


@dataclass(frozen=True)
class PrivateChoiceTarget(Record):
    question_id: str
    correct_option_id: str
    private_note: str

    def __post_init__(self):
        text(self.question_id, "question_id")
        text(self.correct_option_id, "correct_option_id")
        require(type(self.private_note) is str, "Private note must be text")


@dataclass(frozen=True)
class Prediction(Record):
    question_id: str
    answer: str
    selected_option_id: str | None
    abstained: bool
    citations: tuple[str, ...]

    _sequences = {"citations": str}

    def __post_init__(self):
        text(self.question_id, "question_id")
        require(type(self.answer) is str, "Answer must be text")
        require(type(self.abstained) is bool, "abstained must be boolean")
        if self.selected_option_id is not None:
            text(self.selected_option_id, "selected_option_id")
        tuple_of(self.citations, str, "citations")
        require(all(re.fullmatch(r"E[1-9][0-9]*", item) for item in self.citations), "Invalid citation ID")
        require(len(set(self.citations)) == len(self.citations), "Duplicate citations")
        require(not self.abstained or self.selected_option_id is None, "Abstention cannot select an option")


@dataclass(frozen=True)
class ChoiceJudgement(Record):
    question_id: str
    score: float | None
    status: str
    scorer_version: str

    def __post_init__(self):
        text(self.question_id, "question_id")
        text(self.scorer_version, "scorer_version")
        require(self.status in ("ok", "invalid_prediction"), "Invalid judgement status")
        if self.status == "ok":
            require(type(self.score) is float and self.score in (0.0, 1.0), "Invalid choice score")
        else:
            require(self.score is None, "Invalid prediction has no score")


@dataclass(frozen=True)
class CoreCase(Record):
    schema_version: str
    history: History
    question: PublicQuestion
    target: PrivateChoiceTarget
    prediction: Prediction

    _nested = {"history": History, "question": PublicQuestion, "target": PrivateChoiceTarget, "prediction": Prediction}

    def __post_init__(self):
        require(self.schema_version == VERSION, "Unsupported fixture schema")
        require(
            type(self.history) is History
            and type(self.question) is PublicQuestion
            and type(self.target) is PrivateChoiceTarget
            and type(self.prediction) is Prediction,
            "Invalid case objects",
        )
        question = self.question
        require(
            question.history_id == self.history.history_id
            and question.family_id == self.history.family_id
            and question.dataset_id == self.history.dataset_id,
            "History identity mismatch",
        )
        require(question.prefix_sha256 == self.history.prefix_hash(question.last_message_id), "Wrong allowed prefix")
        require(question.question_id == self.target.question_id == self.prediction.question_id, "Question ID mismatch")
        require(question.task_type == "choice", "Core fixture runner supports choice tasks only")
        require(
            self.target.correct_option_id in {item.option_id for item in question.options}, "Invalid gold option ID"
        )


def public_ingest(history, question):
    """Return public canonical messages only, before any HTTP role rendering."""
    require(type(history) is History and type(question) is PublicQuestion, "Expected public input types")
    require(
        (history.history_id, history.family_id, history.dataset_id)
        == (question.history_id, question.family_id, question.dataset_id),
        "History identity mismatch",
    )
    require(question.prefix_sha256 == history.prefix_hash(question.last_message_id), "Prefix hash mismatch")
    return {
        "question_id": question.question_id,
        "history_id": history.history_id,
        "prefix_sha256": question.prefix_sha256,
        "messages": [item.to_dict() for item in history.public_prefix(question.last_message_id)],
    }


def score_choice(question, target, prediction):
    require(
        type(question) is PublicQuestion and type(target) is PrivateChoiceTarget and type(prediction) is Prediction,
        "Expected typed choice inputs",
    )
    require(question.task_type == "choice", "Not a choice task")
    require(question.question_id == target.question_id == prediction.question_id, "Question ID mismatch")
    valid_ids = {item.option_id for item in question.options}
    require(target.correct_option_id in valid_ids, "Invalid gold option ID")
    if prediction.abstained:
        score, status = 0.0, "ok"
    elif prediction.selected_option_id not in valid_ids:
        score, status = None, "invalid_prediction"
    else:
        score, status = float(prediction.selected_option_id == target.correct_option_id), "ok"
    return ChoiceJudgement(question.question_id, score, status, "exact-option-id/0.1.0")
