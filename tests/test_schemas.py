import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from memory_benchmark.core import load_case
from memory_benchmark.schemas import (
    CoreCase,
    PublicQuestion,
    SchemaError,
    SourceMessage,
    canonical,
    digest,
    loads,
    public_ingest,
    score_choice,
)

FIXTURE = Path(__file__).parent / "fixtures/core_choice.json"


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.case = load_case(FIXTURE)

    def test_strict_round_trip(self):
        self.assertEqual(CoreCase.from_dict(self.case.to_dict()), self.case)

    def test_unknown_top_level_field_rejected(self):
        raw = self.case.to_dict()
        raw["unexpected"] = True
        with self.assertRaises(SchemaError):
            CoreCase.from_dict(raw)

    def test_gold_field_cannot_enter_public_question(self):
        raw = self.case.question.to_dict()
        raw["correct_answer"] = "opt_teal"
        with self.assertRaises(SchemaError):
            PublicQuestion.from_dict(raw)

    def test_missing_field_rejected(self):
        raw = self.case.history.messages[0].to_dict()
        del raw["speaker_id"]
        with self.assertRaises(SchemaError):
            SourceMessage.from_dict(raw)

    def test_content_hash_checked(self):
        with self.assertRaises(SchemaError):
            replace(self.case.history.messages[1], content="changed")

    def test_boolean_is_not_an_order(self):
        with self.assertRaises(SchemaError):
            replace(self.case.history.messages[1], source_order=True)

    def test_duplicate_and_reordered_history_rejected(self):
        with self.assertRaises(SchemaError):
            replace(self.case.history, messages=(self.case.history.messages[1],) * 2)
        with self.assertRaises(SchemaError):
            replace(self.case.history, messages=tuple(reversed(self.case.history.messages)))

    def test_history_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.case.history.history_id = "other"

    def test_two_humans_keep_identity_and_role(self):
        messages = public_ingest(self.case.history, self.case.question)["messages"]
        self.assertEqual([item["speaker_id"] for item in messages], ["Alice", "Bob", "Alice"])
        self.assertTrue(all(item["speaker_type"] == "human" for item in messages))

    def test_hidden_target_and_future_not_in_public_projection(self):
        payload = canonical(public_ingest(self.case.history, self.case.question))
        for forbidden in (
            "HIDDEN_PERSONA_DO_NOT_EXPORT",
            "PRIVATE_TARGET_DO_NOT_EXPORT",
            "FUTURE_UPDATE",
            "correct_option_id",
            "private_note",
        ):
            self.assertNotIn(forbidden, payload)
        self.assertIn("I prefer teal notebooks.", payload)

    def test_unknown_time_stays_unknown(self):
        self.assertTrue(
            all(
                item["source_time_raw"] is None
                for item in public_ingest(self.case.history, self.case.question)["messages"]
            )
        )

    def test_wrong_prefix_and_family_rejected(self):
        wrong = replace(self.case.question, last_message_id="m4")
        with self.assertRaises(SchemaError):
            public_ingest(self.case.history, wrong)
        wrong = replace(self.case.question, family_id="other-family")
        with self.assertRaises(SchemaError):
            public_ingest(self.case.history, wrong)

    def test_unclassified_system_refused(self):
        changed = replace(self.case.history.messages[0], visibility="unclassified_system")
        history = replace(self.case.history, messages=(changed, *self.case.history.messages[1:]))
        with self.assertRaises(SchemaError):
            history.public_prefix("m3")

    def test_unknown_cutoff_refused(self):
        with self.assertRaises(SchemaError):
            self.case.history.public_prefix("missing")

    def test_duplicate_options_rejected(self):
        with self.assertRaises(SchemaError):
            replace(self.case.question, options=(self.case.question.options[0],) * 2)

    def test_list_instead_of_immutable_messages_rejected(self):
        with self.assertRaises(SchemaError):
            replace(self.case.history, messages=list(self.case.history.messages))

    def test_correct_wrong_and_abstained_choice(self):
        self.assertEqual(score_choice(self.case.question, self.case.target, self.case.prediction).score, 1.0)
        wrong = replace(self.case.prediction, selected_option_id="opt_red")
        self.assertEqual(score_choice(self.case.question, self.case.target, wrong).score, 0.0)
        abstain = replace(self.case.prediction, selected_option_id=None, abstained=True)
        self.assertEqual(score_choice(self.case.question, self.case.target, abstain).score, 0.0)

    def test_invalid_selection_has_no_score(self):
        invalid = replace(self.case.prediction, selected_option_id="not-an-option")
        result = score_choice(self.case.question, self.case.target, invalid)
        self.assertIsNone(result.score)
        self.assertEqual(result.status, "invalid_prediction")

    def test_mismatched_question_id_refused(self):
        with self.assertRaises(SchemaError):
            score_choice(self.case.question, self.case.target, replace(self.case.prediction, question_id="other"))

    def test_prediction_validation(self):
        for changes in (
            {"abstained": "false"},
            {"abstained": True},
            {"citations": ("E0",)},
            {"citations": ("E1", "E1")},
        ):
            with self.subTest(changes=changes), self.assertRaises(SchemaError):
                replace(self.case.prediction, **changes)

    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        for value in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(value=value), self.assertRaises(SchemaError):
                loads(value)
        with self.assertRaises(SchemaError):
            canonical({"score": float("nan")})

    def test_canonical_hash_is_key_order_independent(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))

    def test_fixture_source_hash_matches_actual_source(self):
        import hashlib

        source_hash = hashlib.sha256(FIXTURE.with_name("source_history.json").read_bytes()).hexdigest()
        self.assertTrue(all(item.source_locator.file_sha256 == source_hash for item in self.case.history.messages))
