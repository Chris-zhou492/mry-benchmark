"""Run public text-memory datasets against the challenge Add/Search API."""

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from itertools import chain, islice
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
SCRIPTMEM_SOURCES = ("angry", "enemy", "friends", "man_earth")
SCRIPTMEM_QUESTIONS = DATASETS / "bundled" / "scriptmem" / "questions.jsonl"
SCRIPTMEM_CONVERSATIONS = DATASETS / "bundled" / "scriptmem" / "conversations.jsonl"
SCRIPTMEM_SOURCE_MANIFEST = DATASETS / "downloads" / "scriptmem" / "source-manifest.json"


def dataset_file(dataset_key: str) -> Path:
    """Resolve a pinned manifest file from the bundled tree or download cache."""
    manifest_path = DATASETS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_key not in manifest["datasets"]:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    entry = manifest["datasets"][dataset_key]["files"][0]
    bundle = entry.get("bundle")
    if bundle:
        path = DATASETS / "bundled" / Path(*bundle.split("/"))
    else:
        relative = entry.get("download_path", f"{dataset_key}/{entry['path']}")
        path = DATASETS / "downloads" / Path(*relative.split("/"))
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {dataset_key} data file: {path.relative_to(ROOT)}. "
            f"Run `python tools/manage_datasets.py download {dataset_key}` first."
        )
    return path


@dataclass
class Session:
    source_id: str
    messages: list[dict[str, Any]]
    source_refs: list[str] = field(default_factory=list)


@dataclass
class Question:
    question_id: str
    query: str
    answer: Any = None
    options: list[str] | None = None
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample:
    dataset: str
    sample_id: str
    sessions: list[Session]
    questions: list[Question]
    metadata: dict[str, Any] = field(default_factory=dict)


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without loading long conversations at once."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as source:
        buffer = ""
        started = False
        ended = False
        eof = False
        while not ended:
            if not eof and len(buffer) < 65536:
                chunk = source.read(65536)
                eof = not chunk
                buffer += chunk
            buffer = buffer.lstrip()
            if not started:
                if not buffer and not eof:
                    continue
                if not buffer.startswith("["):
                    raise ValueError(f"Expected JSON array in {path}")
                buffer = buffer[1:]
                started = True
                continue
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                ended = True
                continue
            if not buffer:
                if eof:
                    raise ValueError(f"Truncated JSON array in {path}")
                continue
            try:
                item, offset = decoder.raw_decode(buffer)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(f"Invalid JSON in {path}: {error}") from error
                chunk = source.read(65536)
                eof = not chunk
                buffer += chunk
                continue
            buffer = buffer[offset:]
            yield item


def _messages(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    messages = []
    refs = []
    for item in raw:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        message = {"role": role, "content": content}
        if isinstance(item.get("timestamp"), int):
            message["timestamp"] = item["timestamp"]
        messages.append(message)
        refs.append(str(item.get("source_ref", "")))
    return messages, refs


def load_longmemeval(variant: str = "s") -> Iterator[Sample]:
    variant = variant.lower()
    dataset_key = {"s": "longmemeval-s", "m": "longmemeval-m", "oracle": "longmemeval-oracle"}.get(variant)
    if dataset_key is None:
        raise ValueError("LongMemEval variant must be s, m, or oracle")
    path = dataset_file(dataset_key)
    for row in iter_json_array(path):
        sessions = []
        for session_id, turns in zip(row["haystack_session_ids"], row["haystack_sessions"], strict=True):
            messages, _ = _messages(turns)
            if messages:
                sessions.append(Session(str(session_id), messages, [str(session_id)] * len(messages)))
        question = Question(
            question_id=str(row["question_id"]),
            query=row["question"],
            answer=row.get("answer"),
            evidence_refs=[str(value) for value in row.get("answer_session_ids", [])],
            metadata={"question_type": row.get("question_type")},
        )
        yield Sample("longmemeval", question.question_id, sessions, [question])


def load_locomo() -> Iterator[Sample]:
    path = dataset_file("locomo")
    for row in iter_json_array(path):
        conversation = row["conversation"]
        speaker_a = conversation["speaker_a"]
        sessions = []
        session_keys = sorted(
            (key for key in conversation if re.fullmatch(r"session_\d+", key)),
            key=lambda key: int(key.split("_")[1]),
        )
        for key in session_keys:
            messages = []
            refs = []
            for turn in conversation[key]:
                text = turn.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                speaker = turn.get("speaker", "")
                messages.append(
                    {
                        "role": "user" if speaker == speaker_a else "assistant",
                        "content": (
                            f"[source_date: {conversation.get(key + '_date_time', 'unknown')}; "
                            f"source_ref: {turn.get('dia_id', '')}] [{speaker}] {text}"
                        ),
                    }
                )
                refs.append(str(turn.get("dia_id", "")))
            if messages:
                sessions.append(Session(key, messages, refs))
        questions = [
            Question(
                question_id=f"{row['sample_id']}:{index}",
                query=qa["question"],
                answer=qa.get("answer", qa.get("adversarial_answer")),
                evidence_refs=[str(value) for value in qa.get("evidence", [])],
                metadata={"category": qa.get("category")},
            )
            for index, qa in enumerate(row["qa"])
        ]
        yield Sample("locomo", str(row["sample_id"]), sessions, questions)


def load_personamem(variant: str = "32k") -> Iterator[Sample]:
    if variant.lower() != "32k":
        raise ValueError("This release currently integrates PersonaMem v1 32k; other variants are not adapted")
    questions_path = dataset_file("personamem-v1-32k")
    contexts_path = questions_path.with_name("shared_contexts_32k.jsonl")
    with questions_path.open(encoding="utf-8-sig", newline="") as source:
        questions = list(csv.DictReader(source))
    contexts: dict[str, list[dict[str, Any]]] = {}
    wanted = {question["shared_context_id"] for question in questions}
    with contexts_path.open(encoding="utf-8") as source:
        for line in source:
            for context_id, context in json.loads(line).items():
                if context_id in wanted:
                    contexts[context_id] = context
            if len(contexts) == len(wanted):
                break
    for row in questions:
        context_id = row["shared_context_id"]
        if context_id not in contexts:
            raise ValueError(f"Missing PersonaMem context: {context_id}")
        end_index = int(row["end_index_in_shared_context"])
        # The system message contains the hidden persona; never add it as memory.
        messages, _ = _messages(contexts[context_id][: end_index + 1])
        options = ast.literal_eval(row["all_options"])
        if not isinstance(options, list):
            raise ValueError(f"Invalid PersonaMem options for {row['question_id']}")
        question = Question(
            question_id=row["question_id"],
            query=row["user_question_or_message"],
            answer=row["correct_answer"],
            options=[str(option) for option in options],
            metadata={"question_type": row["question_type"], "topic": row["topic"], "all_options": row["all_options"]},
        )
        yield Sample("personamem", question.question_id, [Session(context_id, messages)], [question])


def load_clbench() -> Iterator[Sample]:
    path = dataset_file("cl-bench-life")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            turns = row["messages"]
            if not turns or turns[-1].get("role") != "user":
                raise ValueError(f"CL-bench Life row {line_number} has no final user question")
            messages, _ = _messages(turns[:-1])
            sample_id = str(row.get("metadata", {}).get("task_id") or line_number)
            question = Question(
                question_id=sample_id,
                query=turns[-1]["content"],
                metadata={
                    "rubrics": row.get("rubrics", []),
                    "system_prompt": next(
                        (
                            turn["content"]
                            for turn in turns
                            if turn.get("role") == "system" and isinstance(turn.get("content"), str)
                        ),
                        "",
                    ),
                    "qa_type": row.get("qa_type"),
                    "options": row.get("options"),
                    **row.get("metadata", {}),
                },
            )
            yield Sample("clbench-life", sample_id, [Session("history", messages)], [question])


def _beam_questions(raw: Any, sample_id: str) -> list[Question]:
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    groups = raw.items() if isinstance(raw, dict) else [("probe", raw)]
    questions = []
    for category, group in groups:
        items = group if isinstance(group, list) else [group]
        for item in items:
            if not isinstance(item, dict):
                continue
            query = item.get("question") or item.get("query") or item.get("prompt")
            if isinstance(query, str) and query.strip():
                questions.append(
                    Question(
                        question_id=f"{sample_id}:{len(questions)}",
                        query=query,
                        answer=item.get("answer") or item.get("ground_truth"),
                        metadata={
                            "category": category,
                            **{key: item[key] for key in ("rubric_nuggets", "rubrics", "rubric") if key in item},
                        },
                    )
                )
    if not questions:
        raise ValueError(f"Could not parse BEAM probing questions for {sample_id}")
    return questions


def load_beam(variant: str = "100K") -> Iterator[Sample]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("BEAM requires optional pyarrow (install it in the evaluation environment)") from error
    dataset_key = f"beam-{variant.lower()}"
    path = dataset_file(dataset_key)
    file = parquet.ParquetFile(path)
    for batch in file.iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        sample_id = str(row["conversation_id"])
        chat = row["chat"]
        if chat and isinstance(chat[0], list):
            chat = [turn for group in chat for turn in group]
        messages, _ = _messages(chat)
        yield Sample(
            "beam", sample_id, [Session("chat", messages)], _beam_questions(row["probing_questions"], sample_id)
        )


def _scriptmem_sessions(conversation: Any, source: str) -> tuple[list[Session], int, int]:
    if not isinstance(conversation, dict) or "format_example" in conversation:
        raise ValueError(f"ScriptMem {source} conversation is missing or only contains the published schema example")

    session_keys = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.split("_")[1]),
    )
    sessions = []
    skipped_empty = 0
    unspecified_type = 0
    for key in session_keys:
        turns = conversation[key]
        if not isinstance(turns, list):
            raise ValueError(f"ScriptMem {source} {key} must be a list")
        messages = []
        refs = []
        date = str(conversation.get(key + "_date_time", "Unknown"))
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            kind = turn.get("type")
            if not isinstance(kind, str) or not kind.strip():
                kind = "unspecified"
                unspecified_type += 1
            content = turn.get("text")
            if not isinstance(content, str) or not content.strip():
                skipped_empty += 1
                continue
            speaker = turn.get("speaker")
            label = str(speaker) if speaker and str(speaker).lower() != "null" else kind.capitalize()
            source_ref = str(turn.get("dia_id", ""))
            messages.append(
                {
                    # Keep fictional dialogue out of user-profile extraction.
                    "role": "assistant",
                    "content": (
                        f"[source_date: {date}; source_ref: {source_ref}] "
                        f"[{label}] [source_type: {kind}] {content}"
                    ),
                }
            )
            refs.append(source_ref)
        if messages:
            sessions.append(Session(key, messages, refs))
    return sessions, skipped_empty, unspecified_type


def load_scriptmem(source: str | None = None, conversations_path: Path | None = None) -> Iterator[Sample]:
    """Load the public QA package; actual ScriptMem source dialogue is optional and separate."""
    if source is not None and source not in SCRIPTMEM_SOURCES:
        raise ValueError(f"Unknown ScriptMem source: {source}")
    conversations_path = conversations_path or SCRIPTMEM_CONVERSATIONS
    if not conversations_path.is_file():
        conversations_path = None
    source_manifest_sha256 = (
        hashlib.sha256(SCRIPTMEM_SOURCE_MANIFEST.read_bytes()).hexdigest()
        if SCRIPTMEM_SOURCE_MANIFEST.is_file()
        else None
    )

    conversations: dict[tuple[str, str], Any] = {}
    if conversations_path is not None:
        with conversations_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                item_source = str(record.get("dataset") or record.get("source") or "")
                sample_id = str(record.get("sample_id") or "")
                key = (item_source, sample_id)
                if not item_source or not sample_id or key in conversations:
                    raise ValueError(f"Invalid or duplicate ScriptMem conversation key at line {line_number}")
                conversations[key] = record.get("conversation")

    for item_source in ([source] if source else SCRIPTMEM_SOURCES):
        rows: dict[str, list[dict[str, Any]]] = {}
        with SCRIPTMEM_QUESTIONS.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                qa = json.loads(line)
                if qa.get("source") == item_source:
                    rows.setdefault(str(qa.get("sample_id") or ""), []).append(qa)
        for sample_id, sample_questions in rows.items():
            key = (item_source, sample_id)
            conversation = conversations.get(key)
            sessions, skipped_empty, unspecified_type = [], 0, 0
            history_error = "No source conversations were supplied"
            if conversation is not None:
                try:
                    sessions, skipped_empty, unspecified_type = _scriptmem_sessions(conversation, item_source)
                    history_error = ""
                except ValueError as error:
                    history_error = str(error)
            questions = [
                Question(
                    question_id=str(qa.get("qa_id") or qa.get("question_id")),
                    query=qa["question"],
                    answer=qa.get("answer_letters") or qa.get("answer"),
                    options=[str(option) for option in qa.get("option", qa.get("options", []))],
                    metadata={"qa_type": qa.get("qa_type"), "scriptmem_source": item_source},
                )
                for qa in sample_questions
            ]
            yield Sample(
                "scriptmem",
                f"{item_source}:{sample_id}",
                sessions,
                questions,
                {
                    "scriptmem_source": item_source,
                    "skipped_empty_turns": skipped_empty,
                    "unspecified_type_turns": unspecified_type,
                    "source_manifest_sha256": source_manifest_sha256,
                    "conversation_file_sha256": (
                        hashlib.sha256(conversations_path.read_bytes()).hexdigest()
                        if conversations_path is not None
                        else None
                    ),
                    "history_available": bool(sessions),
                    "history_error": history_error,
                },
            )


def chunk_messages(messages: list[dict], chunk_size: int = 6000) -> list[dict]:
    """Compatibility helper: chunk_size is a soft target, never a fixed cut width."""
    from memory_service.record_index import split_documents

    valid = [
        message for message in messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str) and message["content"].strip()
    ]
    return split_documents(valid, target_chars=chunk_size)[0]


def load_samples(
    name: str, variant: str | None = None, scriptmem_conversations_path: Path | None = None
) -> Iterator[Sample]:
    if name == "longmemeval":
        yield from load_longmemeval(variant or "s")
    elif name == "locomo":
        yield from load_locomo()
    elif name == "personamem":
        yield from load_personamem(variant or "32k")
    elif name == "clbench-life":
        yield from load_clbench()
    elif name == "beam":
        yield from load_beam(variant or "100K")
    elif name == "scriptmem":
        yield from load_scriptmem(variant, scriptmem_conversations_path)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def memory_id(request_id: str, index: int) -> str:
    return "mem_" + hashlib.sha256(f"{request_id}:{index}".encode()).hexdigest()[:24]


def prepare_adds(sample: Sample, batch_size: int, run_id: str) -> tuple[str, list[dict[str, Any]], dict[str, set[str]]]:
    user_id = f"public-eval:{run_id}:{sample.dataset}:{sample.sample_id}"
    adds = []
    source_ids: dict[str, set[str]] = {}
    for session in sample.sessions:
        for start in range(0, len(session.messages), batch_size):
            messages = session.messages[start : start + batch_size]
            request_id = f"{user_id}:{session.source_id}:{start // batch_size}"
            adds.append(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "session_id": f"{user_id}:{session.source_id}",
                    "messages": messages,
                }
            )
            for index, source_ref in enumerate(session.source_refs[start : start + batch_size]):
                if source_ref:
                    source_ids.setdefault(source_ref, set()).add(memory_id(request_id, index))
    return user_id, adds, source_ids


def evidence_recall(question: Question, source_ids: dict[str, set[str]], returned_ids: set[str]) -> float | None:
    if not question.evidence_refs:
        return None
    if not any(ref in source_ids for ref in question.evidence_refs):
        return None
    return sum(bool(source_ids.get(ref, set()) & returned_ids) for ref in question.evidence_refs) / len(
        question.evidence_refs
    )


def post_json(
    url: str, payload: dict[str, Any], api_key: str | None, timeout: float, *, bearer: bool = False
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        if bearer:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["X-Api-Key"] = api_key
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(500).decode("utf-8", errors="replace")
            if error.code not in {429, 502, 503, 504} or attempt == 2:
                raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
        except (URLError, TimeoutError) as error:
            if attempt == 2:
                raise RuntimeError(f"Request failed for {url}: {error}") from error
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.top_k < 1 or args.limit < 1 or args.max_questions < 1:
        raise ValueError("batch-size, top-k, limit, and max-questions must be positive")
    api_key = os.environ.get(args.api_key_env)
    if not args.dry_run and not api_key and not args.allow_unauthenticated:
        raise ValueError(f"Set {args.api_key_env} or pass --allow-unauthenticated for a local test")
    conversations_path = getattr(args, "scriptmem_conversations_path", SCRIPTMEM_CONVERSATIONS)
    samples = islice(load_samples(args.dataset, args.variant, conversations_path), args.limit)
    first_sample = next(samples, None)
    if first_sample is None:
        raise ValueError(f"No samples found for {args.dataset}")
    output = Path(args.output)
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.append:
            raise FileExistsError(f"Output exists: {output}; use --append or another path")
    totals = {"samples": 0, "adds": 0, "questions": 0, "evidence_scored": 0, "recall_sum": 0.0}
    sink = output.open("a" if args.append else "w", encoding="utf-8") if not args.dry_run else None
    try:
        for sample in chain((first_sample,), samples):
            user_id, adds, source_ids = prepare_adds(sample, args.batch_size, args.run_id)
            questions = sample.questions[: args.max_questions]
            if not questions or (not adds and not (args.dry_run and sample.dataset == "scriptmem")):
                raise ValueError(f"Sample {sample.dataset}/{sample.sample_id} has no Add messages or test questions")
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "dataset": sample.dataset,
                            "sample_id": sample.sample_id,
                            "sessions": len(sample.sessions),
                            "adds": len(adds),
                            "messages": sum(len(item["messages"]) for item in adds),
                            "questions": len(questions),
                            "evidence_mapped": len(source_ids),
                            "history_available": bool(adds),
                            "source_metadata": sample.metadata,
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                for payload in adds:
                    response = post_json(f"{args.base_url.rstrip('/')}/add", payload, api_key, args.timeout)
                    if response.get("success") is not True or response.get("request_id") != payload["request_id"]:
                        raise RuntimeError(f"Invalid Add response for {payload['request_id']}")
                for question in questions:
                    payload = {"query": question.query, "user_id": user_id, "top_k": args.top_k}
                    if question.options:
                        payload["options"] = question.options
                    response = post_json(f"{args.base_url.rstrip('/')}/search", payload, api_key, args.timeout)
                    results = response.get("data")
                    if not isinstance(results, list) or len(results) > args.top_k:
                        raise RuntimeError(f"Invalid Search response for {question.question_id}")
                    recall = evidence_recall(question, source_ids, {str(item.get("id")) for item in results})
                    record = {
                        "dataset": sample.dataset,
                        "sample_id": sample.sample_id,
                        "question_id": question.question_id,
                        "user_id": user_id,
                        "query": question.query,
                        "answer": question.answer,
                        "options": question.options,
                        "evidence_refs": question.evidence_refs,
                        "evidence_recall_at_k": recall,
                        "metadata": question.metadata,
                        "results": results,
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                    if recall is not None:
                        totals["evidence_scored"] += 1
                        totals["recall_sum"] += recall
            totals["samples"] += 1
            totals["adds"] += len(adds)
            totals["questions"] += len(questions)
    finally:
        if sink:
            sink.close()
    summary = {key: value for key, value in totals.items() if key != "recall_sum"}
    summary["mean_evidence_recall_at_k"] = (
        totals["recall_sum"] / totals["evidence_scored"] if totals["evidence_scored"] else None
    )
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["longmemeval", "locomo", "personamem", "clbench-life", "beam", "scriptmem"],
    )
    parser.add_argument(
        "--variant",
        help="LongMemEval s/m, PersonaMem 32k/128k/1M, BEAM 100K/500K/1M, or ScriptMem source name",
    )
    parser.add_argument(
        "--scriptmem-conversations",
        dest="scriptmem_conversations_path",
        type=Path,
        default=SCRIPTMEM_CONVERSATIONS,
        help="Add-only ScriptMem conversations JSONL (no QA fields)",
    )
    parser.add_argument("--limit", type=int, default=1, help="Maximum samples; a conversation may have many questions")
    parser.add_argument("--max-questions", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50, help="Messages per Add request within a session")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-id", default="dev", help="Namespace for isolated user IDs and idempotent retries")
    parser.add_argument("--base-url", default="http://localhost:8899")
    parser.add_argument("--api-key-env", default="CHALLENGE_API_KEY")
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default=str(Path(__file__).parent / "outputs" / "text_memory.jsonl"))
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Inspect sample sizes without API calls or output writes"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
