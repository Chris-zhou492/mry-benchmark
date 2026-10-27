"""Adapt local Add/Search output to public pipeline inputs, not official scores."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = {"longmemeval", "locomo", "personamem", "clbench-life", "beam", "scriptmem"}


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid {key}")
    return value


def _results(record: dict[str, Any]) -> list[dict[str, Any]]:
    results = record.get("results")
    if not isinstance(results, list):
        raise ValueError("missing or invalid results list")
    selected = []
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("content"), str):
            raise ValueError("search result must have text content")
        if result["content"].strip():
            selected.append(result)
    return selected


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("runner record must be an object")
    dataset = _required_text(record, "dataset")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    metadata = record.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    selected = _results(record)
    context = "\n".join(result["content"] for result in selected)
    normalized = {
        "id": _required_text(record, "question_id"),
        "question": _required_text(record, "query"),
    }

    if dataset in {"longmemeval", "locomo"}:
        if record.get("answer") is None:
            raise ValueError(f"{dataset} record has no reference answer")
        normalized.update(retrieved_context=context, gold_answer=record["answer"])
    elif dataset == "scriptmem":
        if metadata.get("qa_type") not in {"single_choice", "multi_select", "ordering"}:
            raise ValueError("ScriptMem record has no supported qa_type")
        if not isinstance(record.get("options"), list) or not record["options"]:
            raise ValueError("ScriptMem record has no option list")
        normalized.update(
            speaker_1_name="script context",
            speaker_1_memories=context,
            speaker_2_name="additional context",
            speaker_2_memories="",
        )
    elif dataset == "personamem":
        options = metadata.get("all_options")
        if not isinstance(options, str) or not options.strip():
            raise ValueError("PersonaMem v1 requires the original all_options string")
        if record.get("answer") is None:
            raise ValueError("PersonaMem v1 record has no correct answer")
        normalized.update(
            all_options=options,
            correct_answer=record["answer"],
            context_messages=[{"role": "user", "content": result["content"]} for result in selected],
        )
    elif dataset == "clbench-life":
        rubrics = metadata.get("rubrics")
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError("CL-bench record has no rubric list")
        options = record.get("options") or metadata.get("options") or []
        if not isinstance(options, list):
            raise ValueError("CL-bench options must be a list")
        normalized.update(
            rubrics=rubrics,
            system_prompt=str(metadata.get("system_prompt") or ""),
            qa_type=str(metadata.get("qa_type") or ""),
            options=options,
            retrieval={
                "selected": [
                    {"text": result["content"], "created_at": result.get("created_at", "")}
                    for result in selected
                ]
            },
        )
    else:
        rubrics = next(
            (metadata[key] for key in ("rubric_nuggets", "rubrics", "rubric") if isinstance(metadata.get(key), list)),
            None,
        )
        if not rubrics:
            raise ValueError("BEAM record has no rubric list")
        normalized.update(context=context, rubric_nuggets=rubrics)

    return normalized


def convert_file(source: Path, destination: Path) -> tuple[str, int]:
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    dataset: str | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False) as sink:
            temporary = Path(sink.name)
            with source.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        normalized = convert_record(raw)
                    except (json.JSONDecodeError, ValueError) as error:
                        raise ValueError(f"input line {line_number}: {error}") from error
                    if dataset is None:
                        dataset = raw["dataset"]
                    elif raw["dataset"] != dataset:
                        raise ValueError("input must contain only one dataset")
                    sink.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    count += 1
        if not count:
            raise ValueError("input contains no records")
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dataset, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL output from text_memory_runner.py")
    parser.add_argument("--output", type=Path, required=True, help="New normalized JSONL path")
    args = parser.parse_args()
    dataset, count = convert_file(args.input, args.output)
    print(json.dumps({"dataset": dataset, "records": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
