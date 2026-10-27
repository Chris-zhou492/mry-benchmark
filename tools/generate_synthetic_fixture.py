"""Create this project's original synthetic input, with no external data or calls."""

import hashlib
import json
from pathlib import Path

from memory_benchmark.schemas import VERSION, History

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "tests/fixtures"
    destination.mkdir(parents=True, exist_ok=True)
    source = [
        {
            "speaker": "system",
            "type": "system",
            "visibility": "hidden_persona",
            "text": "HIDDEN_PERSONA_DO_NOT_EXPORT: synthetic-only private profile.",
        },
        {"speaker": "Alice", "type": "human", "visibility": "public_history", "text": "I prefer teal notebooks."},
        {"speaker": "Bob", "type": "human", "visibility": "public_history", "text": "I prefer red notebooks."},
        {
            "speaker": "Alice",
            "type": "human",
            "visibility": "public_history",
            "text": "Remember my notebook preference.",
        },
        {
            "speaker": "Alice",
            "type": "human",
            "visibility": "public_history",
            "text": "FUTURE_UPDATE: now I prefer yellow notebooks.",
        },
    ]
    source_text = json.dumps(source, ensure_ascii=False, indent=2) + "\n"
    (destination / "source_history.json").write_text(source_text, encoding="utf-8", newline="\n")
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()
    messages = [
        {
            "message_id": f"m{i}",
            "session_id": "synthetic-session",
            "source_order": i,
            "speaker_id": row["speaker"],
            "speaker_type": row["type"],
            "original_role": row["type"],
            "visibility": row["visibility"],
            "content": row["text"],
            "content_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "source_time_raw": None,
            "source_locator": {"file_sha256": source_hash, "record_pointer": f"/{i}", "message_pointer": "/text"},
        }
        for i, row in enumerate(source)
    ]
    history = History.from_dict(
        {
            "history_id": "synthetic-history",
            "family_id": "synthetic-family",
            "dataset_id": "synthetic-not-a-research-result",
            "loader_version": "synthetic/0.1.0",
            "messages": messages,
        }
    )
    case = {
        "schema_version": VERSION,
        "history": history.to_dict(),
        "question": {
            "question_id": "synthetic-q1",
            "dataset_id": history.dataset_id,
            "history_id": history.history_id,
            "family_id": history.family_id,
            "query": "What notebook color did Alice prefer at this point?",
            "task_type": "choice",
            "last_message_id": "m3",
            "prefix_sha256": history.prefix_hash("m3"),
            "public_instructions": "Choose one option.",
            "options": [
                {"option_id": "opt_red", "content": "Red"},
                {"option_id": "opt_teal", "content": "Teal"},
                {"option_id": "opt_yellow", "content": "Yellow"},
            ],
        },
        "target": {
            "question_id": "synthetic-q1",
            "correct_option_id": "opt_teal",
            "private_note": "PRIVATE_TARGET_DO_NOT_EXPORT: Alice, not Bob; before the update.",
        },
        "prediction": {
            "question_id": "synthetic-q1",
            "answer": "Teal",
            "selected_option_id": "opt_teal",
            "abstained": False,
            "citations": [],
        },
    }
    (destination / "core_choice.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Generated original synthetic fixtures inside this project.")


if __name__ == "__main__":
    main()
