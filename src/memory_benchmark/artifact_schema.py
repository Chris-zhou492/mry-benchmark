"""Generate the checked-in JSON Schema from the runtime artifact contracts."""

from __future__ import annotations

import types
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints

from . import artifacts
from .schemas import Record

_ENUMS = {
    ("TimeInfo", "precision"): ["unknown", "year", "month", "day", "minute", "second", "subsecond"],
    ("TimeInfo", "parse_status"): ["exact", "ambiguous", "missing", "invalid"],
    ("SourceMessageArtifact", "speaker_type"): ["human", "assistant", "tool", "system", "unknown"],
    ("SourceMessageArtifact", "visibility"): [
        "public_history",
        "hidden_persona",
        "public_task_instruction",
        "other_system",
        "unclassified_system",
    ],
    ("PublicQuestionArtifact", "task_type"): ["choice", "factual_text", "numeric", "rubric_response", "unanswerable"],
    ("PrivateTargetArtifact", "target_type"): ["choice", "factual_text", "numeric", "rubric_response", "unanswerable"],
    ("PrivateTargetArtifact", "answerability"): ["answerable", "unanswerable", "uncertain"],
    ("RetrievalArtifact", "status"): ["ok", "client_error", "server_error", "timeout", "invalid_response"],
    ("OmittedHit", "reason"): ["budget_exceeded", "invalid_content", "policy_excluded"],
    ("PredictionArtifact", "status"): ["ok", "format_error", "model_error", "timeout", "refused"],
    ("CriterionResult", "decision"): ["met", "unmet", "uncertain"],
    ("JudgementArtifact", "decision"): ["correct", "partially_correct", "incorrect", "unanswerable", "uncertain"],
    ("JudgementArtifact", "status"): ["ok", "judge_uncertain", "judge_failed", "invalid_prediction"],
    ("AnnotationArtifact", "answerability"): ["answerable", "unanswerable", "uncertain"],
    ("AnnotationArtifact", "annotator_status"): ["draft", "reviewed", "adjudicated", "uncertain"],
    ("DiagnosticCheck", "result"): ["met", "unmet", "uncertain", "not_evaluable"],
    ("DiagnosisArtifact", "suspected_stages"): [],
    ("DiagnosisArtifact", "status"): ["ok", "not_evaluable", "diagnostic_failed"],
    ("RunManifest", "status"): ["planned", "validated", "running", "partial", "completed", "failed", "aborted"],
}

_STAGE_ENUM = [
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
]


def _closed(properties, required):
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_NONEMPTY = {"type": "string", "minLength": 1}
_OPTIONAL_TEXT = {"anyOf": [_NONEMPTY, {"type": "null"}]}
_MANIFEST_DEFS = {
    "ManifestMethod": _closed(
        {
            "method_id": _NONEMPTY,
            "method_version": _NONEMPTY,
            "endpoint_id": _NONEMPTY,
            "configuration_sha256": _HASH,
            "provider": _OPTIONAL_TEXT,
            "service_version": _OPTIONAL_TEXT,
            "options": {"type": "object"},
        },
        ["method_id", "method_version", "endpoint_id", "configuration_sha256"],
    ),
    "ManifestReader": _closed(
        {
            "reader_id": _NONEMPTY,
            "reader_version": _NONEMPTY,
            "configuration_sha256": _HASH,
            "status": {"type": "string", "enum": ["configured", "not-executed", "executed", "failed"]},
            "provider": _OPTIONAL_TEXT,
            "requested_model": _OPTIONAL_TEXT,
            "actual_model": _OPTIONAL_TEXT,
            "prompt_sha256": _HASH,
            "model_snapshot": {},
            "options": {"type": "object"},
            "reason": {"type": "string"},
        },
        ["reader_id", "reader_version", "configuration_sha256", "status"],
    ),
    "ManifestScorer": _closed(
        {
            "scorer_id": _NONEMPTY,
            "scorer_version": _NONEMPTY,
            "configuration_sha256": _HASH,
            "status": {"type": "string", "enum": ["configured", "not-executed", "executed", "failed"]},
            "provider": _OPTIONAL_TEXT,
            "requested_model": _OPTIONAL_TEXT,
            "actual_model": _OPTIONAL_TEXT,
            "prompt_sha256": _HASH,
            "options": {"type": "object"},
        },
        ["scorer_id", "scorer_version", "configuration_sha256", "status"],
    ),
    "ManifestEvidencePolicy": _closed(
        {
            "strategy": {"type": "string", "enum": ["whole_hit", "fixed_slice"]},
            "budget_tokens": {"type": "integer", "minimum": 1},
            "tokenizer_id": _NONEMPTY,
            "tokenizer_revision": _NONEMPTY,
            "prompt_reserve_tokens": {"type": "integer", "minimum": 0},
            "output_reserve_tokens": {"type": "integer", "minimum": 0},
            "overflow_policy": _NONEMPTY,
        },
        ["strategy", "budget_tokens", "tokenizer_id", "tokenizer_revision"],
    ),
    "ManifestRetryPolicy": _closed(
        {
            "max_attempts": {"type": "integer", "minimum": 1, "maximum": 10},
            "backoff_seconds": {"type": "number", "minimum": 0},
            "retry_statuses": {"type": "array", "items": {"type": "integer", "minimum": 100, "maximum": 599}},
        },
        ["max_attempts", "backoff_seconds"],
    ),
    "ManifestCachePolicy": _closed(
        {
            "backend_state_verification": {
                "type": "string",
                "enum": ["snapshot", "state_endpoint", "rebuild", "disabled"],
            },
            "raw_responses": {"type": "string", "enum": ["retain", "redact", "not-applicable"]},
            "model_cache": {"type": "boolean"},
            "key_version": _NONEMPTY,
        },
        ["backend_state_verification", "raw_responses"],
    ),
    "ManifestEnvironment": _closed(
        {
            "python": _NONEMPTY,
            "implementation": _NONEMPTY,
            "platform": _OPTIONAL_TEXT,
            "machine": _OPTIONAL_TEXT,
            "dependency_versions": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        ["python", "implementation"],
    ),
}

_HASH_FIELDS = {
    ("SourceFile", "sha256"),
    ("SourceLocator", "file_sha256"),
    ("SourceMessageArtifact", "content_sha256"),
    ("HistoryArtifact", "content_sha256"),
    ("AllowedHistory", "prefix_sha256"),
    ("AddReceipt", "history_prefix_sha256"),
    ("AddReceipt", "payload_sha256"),
    ("EvidenceBundle", "content_sha256"),
    ("PredictionArtifact", "reader_config_hash"),
    ("JudgementArtifact", "target_sha256"),
    ("EvidenceSpan", "quote_sha256"),
    ("RunManifest", "protocol_sha256"),
    ("RunManifest", "split_sha256"),
}


def _type_schema(annotation):
    if annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        non_null = [item for item in args if item is not type(None)]
        schemas = [_type_schema(item) for item in non_null]
        if len(non_null) != len(args):
            schemas.append({"type": "null"})
        return {"anyOf": schemas}
    if origin in (tuple, list):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _type_schema(item_type)}
    if origin is dict:
        return {"type": "object", "additionalProperties": _type_schema(args[1]) if len(args) > 1 else True}
    if is_dataclass(annotation) and issubclass(annotation, Record):
        return {"$ref": f"#/$defs/{annotation.__name__}"}
    if annotation is str:
        return {"type": "string"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    return {}


def build_json_schema() -> dict[str, Any]:
    records = {
        name: value
        for name, value in vars(artifacts).items()
        if isinstance(value, type) and value is not Record and issubclass(value, Record) and is_dataclass(value)
    }
    definitions: dict[str, Any] = {}
    for name, model in records.items():
        hints = get_type_hints(model)
        properties = {}
        for field in fields(model):
            schema = _type_schema(hints[field.name])
            if name == "RunManifest" and field.name == "method":
                schema = {"$ref": "#/$defs/ManifestMethod"}
            elif name == "RunManifest" and field.name == "reader":
                schema = {"$ref": "#/$defs/ManifestReader"}
            elif name == "RunManifest" and field.name == "scorers":
                schema = {"type": "array", "items": {"$ref": "#/$defs/ManifestScorer"}}
            elif name == "RunManifest" and field.name == "evidence_policy":
                schema = {"$ref": "#/$defs/ManifestEvidencePolicy"}
            elif name == "RunManifest" and field.name == "retry_policy":
                schema = {"$ref": "#/$defs/ManifestRetryPolicy"}
            elif name == "RunManifest" and field.name == "cache_policy":
                schema = {"$ref": "#/$defs/ManifestCachePolicy"}
            elif name == "RunManifest" and field.name == "environment":
                schema = {"$ref": "#/$defs/ManifestEnvironment"}
            elif name == "RunManifest" and field.name == "source_file_hashes":
                schema = {"type": "object", "minProperties": 1, "additionalProperties": _HASH}
            elif name == "RunManifest" and field.name == "loader_versions":
                schema = {"type": "object", "additionalProperties": {"type": "string", "minLength": 1}}
            enum = _ENUMS.get((name, field.name))
            if name == "DiagnosisArtifact" and field.name == "suspected_stages":
                schema["items"]["enum"] = _STAGE_ENUM
            elif enum:
                schema["enum"] = enum
            if (name, field.name) in _HASH_FIELDS:
                schema["pattern"] = "^[0-9a-f]{64}$"
            if name == "AddReceipt" and field.name == "acknowledged":
                schema["const"] = True
            if field.name == "schema_version" and name in {
                "HistoryArtifact",
                "PublicQuestionArtifact",
                "PrivateTargetArtifact",
                "AddReceipt",
                "RetrievalArtifact",
                "EvidenceBundle",
                "PredictionArtifact",
                "JudgementArtifact",
                "AnnotationArtifact",
                "DiagnosisArtifact",
            }:
                schema["const"] = artifacts.ARTIFACT_SCHEMA_VERSION
            if name == "HistoryArtifact" and field.name in ("source_files", "messages"):
                schema["minItems"] = 1
            if name == "RunManifest" and field.name == "selected_question_ids":
                schema["minItems"] = 1
            if field.name.endswith("_at") or (name == "TimeInfo" and field.name == "parsed_iso8601"):
                if schema.get("type") == "string":
                    schema["format"] = "date-time"
                elif schema.get("anyOf") and schema["anyOf"][0].get("type") == "string":
                    schema["anyOf"][0]["format"] = "date-time"
            if name == "RunManifest" and field.name == "code_revision":
                schema["pattern"] = "^(git:[0-9a-f]{40,64}|snapshot-sha256:[0-9a-f]{64}|unversioned)$"
            properties[field.name] = schema
        definitions[name] = {
            "type": "object",
            "properties": properties,
            "required": [item.name for item in fields(model)],
            "additionalProperties": False,
        }
    definitions.update(_MANIFEST_DEFS)
    definitions["PublicQuestionArtifact"]["allOf"] = [
        {
            "if": {"properties": {"task_type": {"const": "choice"}}, "required": ["task_type"]},
            "then": {"properties": {"options": {"type": "array", "minItems": 2}}},
        },
        {
            "if": {
                "properties": {"task_type": {"enum": ["factual_text", "numeric", "rubric_response", "unanswerable"]}},
                "required": ["task_type"],
            },
            "then": {"properties": {"options": {"type": "array", "maxItems": 0}}},
        },
    ]
    definitions["PrivateTargetArtifact"]["allOf"] = [
        {
            "if": {"properties": {"target_type": {"const": "choice"}}, "required": ["target_type"]},
            "then": {
                "properties": {
                    "answer": {
                        "type": "object",
                        "properties": {"correct_option_id": _NONEMPTY},
                        "required": ["correct_option_id"],
                        "additionalProperties": False,
                    }
                }
            },
        },
        {
            "if": {"properties": {"target_type": {"const": "rubric_response"}}, "required": ["target_type"]},
            "then": {"properties": {"criteria": {"type": "array", "minItems": 1}}},
        },
    ]
    definitions["TimeInfo"]["allOf"] = [
        {
            "if": {"properties": {"parse_status": {"const": "exact"}}, "required": ["parse_status"]},
            "then": {
                "properties": {
                    "raw": {"type": "string"},
                    "parsed_iso8601": {"type": "string", "format": "date-time"},
                    "timezone": _NONEMPTY,
                    "precision": {"not": {"const": "unknown"}},
                }
            },
        },
        {
            "if": {"properties": {"parse_status": {"const": "missing"}}, "required": ["parse_status"]},
            "then": {
                "properties": {
                    "raw": {"type": "null"},
                    "parsed_iso8601": {"type": "null"},
                    "timezone": {"type": "null"},
                    "precision": {"const": "unknown"},
                }
            },
        },
        {
            "if": {"properties": {"parse_status": {"enum": ["ambiguous", "invalid"]}}, "required": ["parse_status"]},
            "then": {"properties": {"raw": {"type": "string"}, "parsed_iso8601": {"type": "null"}}},
        },
    ]
    definitions["PredictionArtifact"]["allOf"] = [
        {
            "if": {"properties": {"status": {"const": "ok"}}, "required": ["status"]},
            "then": {"properties": {"raw_response_ref": _NONEMPTY}},
        }
    ]
    definitions["SourceMessageArtifact"]["allOf"] = [
        {
            "if": {"properties": {"speaker_type": {"const": "system"}}, "required": ["speaker_type"]},
            "then": {"properties": {"visibility": {"not": {"const": "public_history"}}}},
        }
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:memory-benchmark:artifacts:0.2.0-draft",
        "title": "Memory Benchmark Artifact Contracts",
        "oneOf": [{"$ref": f"#/$defs/{item.__name__}"} for item in artifacts.ROOT_ARTIFACTS],
        "$defs": definitions,
    }
