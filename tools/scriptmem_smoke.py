"""Check the integrity and usable scope of the pinned public ScriptMem package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "datasets" / "bundled" / "scriptmem" / "questions.jsonl"
CONVERSATIONS = ROOT / "datasets" / "bundled" / "scriptmem" / "conversations.jsonl"
REQUIRED_QUESTION_FIELDS = {"question", "answer", "conversation_id", "qa_type"}
EXPECTED_QUESTION_COUNT = 457
EXPECTED_CONVERSATION_COUNT = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def main() -> None:
    questions = read_jsonl(QUESTIONS)
    conversations = read_jsonl(CONVERSATIONS)
    if len(questions) != EXPECTED_QUESTION_COUNT:
        raise SystemExit(f"Expected {EXPECTED_QUESTION_COUNT} questions, found {len(questions)}")
    if len(conversations) != EXPECTED_CONVERSATION_COUNT:
        raise SystemExit(f"Expected {EXPECTED_CONVERSATION_COUNT} conversation records, found {len(conversations)}")
    for index, row in enumerate(questions, start=1):
        missing = REQUIRED_QUESTION_FIELDS - row.keys()
        if missing:
            raise SystemExit(f"Question {index} is missing fields: {', '.join(sorted(missing))}")
        answer = row["answer"]
        if isinstance(answer, str):
            has_answer = bool(answer.strip())
        elif isinstance(answer, list):
            has_answer = any(isinstance(value, str) and value.strip() for value in answer)
        else:
            has_answer = False
        if not has_answer:
            raise SystemExit(f"Question {index} has no usable reference answer")

    has_source_dialogue = any(
        isinstance(row.get("conversation"), str) and len(row["conversation"].strip()) > 100
        for row in conversations
    )
    print(f"ScriptMem data check passed: {len(questions)} QA rows, {len(conversations)} conversation records.")
    if not has_source_dialogue:
        print("Source script conversations are absent; this package supports QA/evaluator checks only, not memory Add/Search.")


if __name__ == "__main__":
    main()
