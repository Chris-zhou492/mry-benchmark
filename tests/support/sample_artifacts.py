"""Small independently authored source objects for mock HTTP tests."""

import hashlib
from pathlib import Path

from memory_benchmark.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    AllowedHistory,
    HistoryArtifact,
    PublicQuestionArtifact,
    SourceFile,
    SourceLocator,
    SourceMessageArtifact,
    TimeInfo,
    history_content_hash,
)
from memory_benchmark.schemas import digest


def make_sample():
    raw_source = (Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_mock_source.json").read_bytes()
    file_hash = hashlib.sha256(raw_source).hexdigest()
    source_file = SourceFile("tests/fixtures/synthetic_mock_source.json", file_hash)

    def message(message_id, order, speaker_id, speaker_type, role, visibility, content, time):
        return SourceMessageArtifact(
            message_id=message_id,
            session_id="session-1",
            source_order=order,
            speaker_id=speaker_id,
            speaker_type=speaker_type,
            original_role=role,
            visibility=visibility,
            content=content,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            source_locator=SourceLocator(file_hash, "/history", f"/history/messages/{order}", None, None),
            time=time,
        )

    missing_time = TimeInfo(None, None, None, "unknown", "missing")
    history_messages = (
        message(
            "hidden-persona",
            0,
            "system",
            "system",
            "system",
            "hidden_persona",
            "HIDDEN_PERSONA_DO_NOT_EXPORT",
            missing_time,
        ),
        message(
            "m1",
            1,
            "Alice",
            "human",
            "user",
            "public_history",
            "My preferred tea is jasmine.",
            TimeInfo("2025-01-01 10:00", "2025-01-01T10:00:00+00:00", "UTC", "minute", "exact"),
        ),
        message("m2", 2, "Bob", "human", "user", "public_history", "I prefer oolong tea.", missing_time),
        message(
            "m3", 3, "Alice", "human", "user", "public_history", "Correction: I now prefer green tea.", missing_time
        ),
        message(
            "m4", 4, "Alice", "human", "user", "public_history", "FUTURE_UPDATE: I now prefer herbal tea.", missing_time
        ),
    )
    source_files = (source_file,)
    history = HistoryArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        history_id="hist-synth-1",
        family_id="family-synth-1",
        dataset_id="synthetic-fixture",
        source_files=source_files,
        messages=history_messages,
        loader_version="synthetic-fixture/0.1.0",
        content_sha256=history_content_hash(source_files, history_messages),
    )
    public_prefix = history_messages[1:4]
    prefix_hash = digest([item.to_dict() for item in public_prefix])
    question = PublicQuestionArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        question_id="q-synth-paired-preferences",
        dataset_id="synthetic-fixture",
        history_id=history.history_id,
        family_id=history.family_id,
        query="tea preferences for Alice and Bob",
        task_type="rubric_response",
        allowed_history=AllowedHistory("m3", 3, prefix_hash),
        public_instructions=("Summarize each person's preference separately.",),
        options=(),
        source_locator=SourceLocator(file_hash, "/questions/0", "/questions/0/query", None, None),
    )
    return history, question, source_file
