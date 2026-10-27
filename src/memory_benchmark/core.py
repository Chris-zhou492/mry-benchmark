"""Offline core acceptance harness, deliberately without HTTP or model calls."""

from pathlib import Path

from .schemas import VERSION, CoreCase, digest, loads, public_ingest, score_choice, text_hash
from .state import PROJECT_ROOT, RunStore


def load_case(path):
    return CoreCase.from_dict(loads(Path(path).read_text(encoding="utf-8")))


def smoke(path, output_root, run_id, *, resume=False):
    case = load_case(path)
    settings = {
        "schema_version": VERSION,
        "mode": "synthetic-core-only",
        "case_sha256": digest(case.to_dict()),
        "scorer": "exact-option-id/0.1.0",
        "reader": "fixture-prediction-no-model",
        "implementation_sha256": digest(
            {
                name: text_hash(Path(__file__).with_name(name).read_text(encoding="utf-8"))
                for name in ("core.py", "schemas.py", "state.py")
            }
        ),
    }
    with RunStore(output_root, run_id, settings, resume=resume).locked() as store:
        public = store.run_step(
            "prepare_public_input",
            {"history": case.history.to_dict(), "question": case.question.to_dict()},
            lambda: public_ingest(case.history, case.question),
        )
        prediction = store.run_step("validate_fixture_prediction", case.prediction.to_dict(), case.prediction.to_dict)
        judgement = store.run_step(
            "score_choice",
            {
                "question": case.question.to_dict(),
                "target_sha256": digest(case.target.to_dict()),
                "prediction": prediction,
            },
            lambda: score_choice(case.question, case.target, case.prediction).to_dict(),
        )
        return {
            "mode": "synthetic-core-only",
            "run_id": run_id,
            "public_message_count": len(public["messages"]),
            "judgement": judgement,
            "model_calls": 0,
            "memory_http_calls": 0,
            "ledger": str(store.path / "ledger.json"),
        }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect-fixture", "smoke"))
    parser.add_argument("--fixture", type=Path, default=PROJECT_ROOT / "tests/fixtures/core_choice.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--run-id", default="synthetic-core-v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect-fixture":
        case = load_case(args.fixture)
        public = public_ingest(case.history, case.question)
        result = {
            "schema_version": VERSION,
            "history_id": case.history.history_id,
            "question_id": case.question.question_id,
            "source_messages": len(case.history.messages),
            "allowed_public_messages": len(public["messages"]),
            "model_calls": 0,
            "memory_http_calls": 0,
        }
    else:
        result = smoke(args.fixture, args.output_root, args.run_id, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
