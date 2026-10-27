import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from memory_benchmark.artifact_schema import build_json_schema
from memory_benchmark.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    AnnotationArtifact,
    Criterion,
    DiagnosisArtifact,
    DiagnosticCheck,
    EvidenceSpan,
    PrivateTargetArtifact,
    RequiredEvidenceSet,
    RunManifest,
    SchemaError,
    SourceLocator,
    TimeInfo,
)
from memory_benchmark.provenance import new_validation_manifest
from tests.support.sample_artifacts import make_sample

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.history, self.question, self.source_file = make_sample()

    def test_history_question_round_trip_and_integrity(self):
        from memory_benchmark.artifacts import HistoryArtifact, PublicQuestionArtifact

        self.assertEqual(HistoryArtifact.from_dict(self.history.to_dict()), self.history)
        self.assertEqual(PublicQuestionArtifact.from_dict(self.question.to_dict()), self.question)
        source_path = PROJECT_ROOT / self.source_file.path
        self.assertEqual(hashlib.sha256(source_path.read_bytes()).hexdigest(), self.source_file.sha256)
        corrupted = self.history.to_dict()
        corrupted["messages"][1]["content"] = "different text"
        with self.assertRaises(SchemaError):
            HistoryArtifact.from_dict(corrupted)
        unknown = self.question.to_dict()
        unknown["gold_answer"] = "private"
        with self.assertRaises(SchemaError):
            PublicQuestionArtifact.from_dict(unknown)

    def test_time_precision_does_not_invent_timestamps(self):
        with self.assertRaises(SchemaError):
            TimeInfo(None, "2025-01-01T00:00:00Z", "UTC", "second", "missing")
        with self.assertRaises(SchemaError):
            TimeInfo("yesterday", "2025-01-01T00:00:00Z", "UTC", "second", "ambiguous")
        self.assertEqual(self.history.messages[2].time.parse_status, "missing")
        self.assertIsNone(self.history.messages[2].time.parsed_iso8601)

    def test_private_target_rubric_annotation_and_diagnosis_round_trip(self):
        locator = SourceLocator(self.source_file.sha256, "/questions/0", "/questions/0/target", None, None)
        target = PrivateTargetArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            question_id=self.question.question_id,
            target_type="rubric_response",
            answer={"expected": "Alice: green tea; Bob: oolong tea"},
            acceptable_answers=({"alice": "green tea", "bob": "oolong tea"},),
            criteria=(
                Criterion("alice-current", "Names Alice's current preference", 1.0, "synthetic annotation"),
                Criterion("bob-current", "Names Bob's preference", 1.0, "synthetic annotation"),
            ),
            numeric_policy={},
            answerability="answerable",
            origin=locator,
        )
        self.assertEqual(PrivateTargetArtifact.from_dict(target.to_dict()), target)
        choice_target = PrivateTargetArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            question_id="q-choice",
            target_type="choice",
            answer={"correct_option_id": "option-green"},
            acceptable_answers=(),
            criteria=(),
            numeric_policy={},
            answerability="answerable",
            origin=locator,
        )
        self.assertEqual(PrivateTargetArtifact.from_dict(choice_target.to_dict()), choice_target)
        with self.assertRaises(SchemaError):
            replace(choice_target, answer={"text": "green"})
        span = EvidenceSpan(
            "m3",
            0,
            len("Correction: I now prefer green tea."),
            hashlib.sha256(b"Correction: I now prefer green tea.").hexdigest(),
        )
        annotation = AnnotationArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            annotation_id="ann-synth-1",
            question_id=self.question.question_id,
            source_revision=self.source_file.sha256,
            required_evidence_sets=(
                RequiredEvidenceSet("alice-update", (span,)),
                RequiredEvidenceSet(
                    "bob-fact", (EvidenceSpan("m2", 0, 20, hashlib.sha256(b"I prefer oolong tea.").hexdigest()),)
                ),
            ),
            claims=(
                {"claim_id": "alice-current", "value": "green tea"},
                {"claim_id": "bob-current", "value": "oolong tea"},
            ),
            temporal_scope={"as_of_message_id": "m3", "updates_supersede_earlier": True},
            state_relations=({"from": "m1", "to": "m3", "relation": "supersedes"},),
            answerability="answerable",
            annotator_status="draft",
            adjudication_ref=None,
            version="synthetic-annotation/0.1.0",
        )
        self.assertEqual(AnnotationArtifact.from_dict(annotation.to_dict()), annotation)
        diagnosis = DiagnosisArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            diagnosis_id="diag-synth-1",
            question_id=self.question.question_id,
            retrieval_id="ret-synth-1",
            prediction_id="pred-synth-1",
            annotation_version=annotation.version,
            checks=(
                DiagnosticCheck(
                    "evidence-set-coverage", "uncertain", "Mock retrieval only; no semantic adjudication", ("E1", "E2")
                ),
            ),
            suspected_stages=("unknown",),
            confidence_basis="No human or model judgment executed in the HTTP acceptance.",
            intervention_refs=(),
            status="not_evaluable",
        )
        self.assertEqual(DiagnosisArtifact.from_dict(diagnosis.to_dict()), diagnosis)

    def test_manifest_records_snapshot_and_refuses_credentials(self):
        manifest = new_validation_manifest(
            run_id="manifest-synth",
            question_ids=(self.question.question_id,),
            method_id="mock-memory",
            endpoint_id="local-mock-v1",
            evidence_budget=100,
            tokenizer_id="synthetic-whitespace",
            tokenizer_revision="0.1.0",
            retry_max_attempts=2,
        )
        self.assertTrue(manifest.code_revision.startswith("snapshot-sha256:"))
        self.assertIn("paper/pipeline_spec.md", manifest.source_file_hashes)
        self.assertEqual(RunManifest.from_dict(manifest.to_dict()), manifest)
        with self.assertRaises(SchemaError):
            replace(manifest, reader={"api_key": "never-store-me"})
        with self.assertRaises(SchemaError):
            replace(manifest, code_revision="made-up-commit")

    def test_checked_in_json_schema_matches_runtime_contracts(self):
        schema_path = PROJECT_ROOT / "protocols" / "artifacts.schema.json"
        checked_in = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(checked_in, build_json_schema())
        self.assertEqual(checked_in["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertGreaterEqual(len(checked_in["$defs"]), 25)
        for name in ("EvidenceBundle", "PrivateTargetArtifact", "RunManifest", "DiagnosisArtifact"):
            self.assertIn(name, checked_in["$defs"])
            self.assertFalse(checked_in["$defs"][name]["additionalProperties"])

        def verify_refs(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    self.assertTrue(value["$ref"].startswith("#/$defs/"))
                    self.assertIn(value["$ref"].split("/")[-1], checked_in["$defs"])
                for nested in value.values():
                    verify_refs(nested)
            elif isinstance(value, list):
                for nested in value:
                    verify_refs(nested)

        verify_refs(checked_in)
