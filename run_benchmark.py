"""Preflight, run and resume public Memory benchmarks with per-question reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from itertools import islice
from pathlib import Path

from dotenv import load_dotenv

import text_memory_runner as runner
from benchmark_state import (
    append_row,
    atomic_json,
    digest,
    index_rows,
    judge_failed,
    read_rows,
    row_id,
    run_lock,
    safe_failure,
    validate_hits,
    validate_url,
)
from prepare_pipeline_inputs import convert_record

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "longmemeval": ("s", "data/longmemeval-s/pipeline.py", "is_correct"),
    "locomo": (None, "data/locomo-refined/pipeline.py", "is_correct"),
    "personamem": ("32k", "data/personamem/pipeline_v1.py", "is_correct"),
    "clbench-life": (None, "data/clbench/pipeline.py", "rubric_clbench_score"),
    "beam": ("100K", "data/beam/pipeline.py", "llm_judge_score"),
    "scriptmem": (None, "data/scriptmem/pipeline.py", "score"),
}
DEFAULT_DATASETS = ["locomo"]


def samples(plan):
    loaded = (
        runner.load_samples(plan.dataset, plan.variant, getattr(plan, "scriptmem_conversations_path", None))
        if plan.dataset == "scriptmem"
        else runner.load_samples(plan.dataset, plan.variant)
    )
    source = islice(loaded, plan.limit)
    if plan.dataset != "scriptmem":
        yield from source
        return

    remaining = plan.max_questions
    selected_types = getattr(plan, "scriptmem_qa_types", None)
    for sample in source:
        if selected_types:
            grouped = {
                qa_type: [question for question in sample.questions if question.metadata.get("qa_type") == qa_type]
                for qa_type in selected_types
            }
            selected = []
            while len(selected) < remaining:
                found_question = False
                for qa_type in selected_types:
                    if grouped[qa_type]:
                        selected.append(grouped[qa_type].pop(0))
                        found_question = True
                        if len(selected) == remaining:
                            break
                if not found_question:
                    break
            selected_ids = {question.question_id for question in selected}
            sample = runner.Sample(
                sample.dataset,
                sample.sample_id,
                sample.sessions,
                [question for question in sample.questions if question.question_id in selected_ids],
                sample.metadata,
            )
        questions = sample.questions[:remaining]
        if questions:
            yield runner.Sample(sample.dataset, sample.sample_id, sample.sessions, questions, sample.metadata)
            remaining -= len(questions)
        if remaining <= 0:
            break


def inspect_plan(plan) -> dict:
    signature = hashlib.sha256()
    identifiers = []
    adds_count = 0
    source_metadata = []
    for sample in samples(plan):
        _, adds, _ = runner.prepare_adds(sample, plan.batch_size, plan.run_id)
        questions = sample.questions[: plan.max_questions]
        if (not adds and plan.dataset != "scriptmem") or not questions:
            raise ValueError(f"Empty sample in {plan.dataset}")
        signature.update(digest([adds, [question.__dict__ for question in questions], sample.metadata]).encode())
        identifiers.extend(question.question_id for question in questions)
        adds_count += len(adds)
        source_metadata.append(sample.metadata)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError(f"No questions or duplicate IDs in {plan.dataset}")
    result = {"data_sha256": signature.hexdigest(), "question_ids": identifiers, "adds": adds_count}
    if plan.dataset == "scriptmem":
        result["source_metadata"] = source_metadata
    print(
        json.dumps(
            {
                "dataset": plan.dataset,
                "questions": len(identifiers),
                "adds": adds_count,
                **({"history_available": adds_count > 0} if plan.dataset == "scriptmem" else {}),
                **({"source_metadata": source_metadata} if plan.dataset == "scriptmem" else {}),
            }
        )
    )
    return result


def source_provenance(name: str, plan) -> dict:
    manifest = json.loads((ROOT / "datasets" / "manifest.json").read_text(encoding="utf-8"))
    keys = {
        "longmemeval": f"longmemeval-{plan.variant}",
        "locomo": "locomo",
        "personamem": "personamem-v1-32k",
        "clbench-life": "cl-bench-life",
        "beam": f"beam-{str(plan.variant).lower()}",
        "scriptmem": "scriptmem",
    }
    dataset_key = keys[name]
    dataset = manifest["datasets"][dataset_key]
    return {
        "dataset_key": dataset_key,
        "revision": dataset["revision"],
        "license": dataset["license"],
        "files": [
            {
                "path": item["path"],
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size"),
                "bundle": item.get("bundle"),
            }
            for item in dataset["files"]
        ],
    }


def summarize(retrieval: Path, scores: Path, field: str) -> dict:
    expected = [str(row["question_id"]) for row in read_rows(retrieval)]
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate question IDs")
    indexed = index_rows(scores, set(expected))
    if set(indexed) != set(expected):
        raise ValueError("Score/retrieval question ID mismatch")
    values = [row.get(field) for row in indexed.values()]
    if not values or any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in values
    ):
        raise ValueError(f"Missing or invalid score field: {field}")
    failed = sum(judge_failed(row) for row in indexed.values())
    valid = [row[field] for row in indexed.values() if not judge_failed(row)]
    return {
        "questions": len(expected),
        "scored_questions": len(valid),
        "judge_failures": failed,
        "score_field": field,
        "mean_score": sum(values) / len(values),
        "mean_score_valid": sum(valid) / len(valid) if valid else None,
    }


def execute(arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-B", *arguments], cwd=ROOT, check=True)


def preflight(args, *, memory: bool, answer: bool, judge: bool) -> None:
    key = os.environ.get(args.api_key_env)
    cache = getattr(args, "_probe_cache", set())
    args._probe_cache = cache
    memory_tag = ("memory", args.base_url, digest(key))
    if memory and memory_tag not in cache:
        response = runner.post_json(
            args.base_url.rstrip("/") + "/search",
            {"user_id": "public-eval:preflight:" + uuid.uuid4().hex, "query": "benchmark readiness probe", "top_k": 1},
            key,
            args.timeout,
        )
        if validate_hits(response, 1):
            raise ValueError("Memory returned records for a fresh probe user; check user isolation")
        cache.add(memory_tag)
    for prefix, needed in [("ANSWER", answer), ("JUDGE", judge)]:
        if not needed:
            continue
        model_tag = (
            "model",
            os.environ[f"{prefix}_API_BASE"].rstrip("/"),
            os.environ[f"{prefix}_MODEL"],
            digest(os.environ[f"{prefix}_API_KEY"]),
        )
        if model_tag in cache:
            continue
        response = runner.post_json(
            os.environ[f"{prefix}_API_BASE"].rstrip("/") + "/chat/completions",
            {
                "model": os.environ[f"{prefix}_MODEL"],
                "messages": [{"role": "user", "content": "benchmark readiness probe: reply OK"}],
                "max_tokens": 32,
                **({"enable_thinking": False} if "dashscope.aliyuncs.com" in os.environ[f"{prefix}_API_BASE"] else {}),
            },
            os.environ[f"{prefix}_API_KEY"],
            args.timeout,
            bearer=True,
        )
        choices = response.get("choices") if isinstance(response, dict) else None
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("message"), dict)
            or not isinstance(choices[0]["message"].get("content"), str)
            or not choices[0]["message"]["content"].strip()
        ):
            raise ValueError(f"{prefix} endpoint returned an invalid chat completion")
        cache.add(model_tag)


def run_retrieval(plan, path: Path, checkpoint: Path, expected: set[str], *, resume: bool) -> None:
    indexed = index_rows(path, expected, repair_tail=resume)
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {"adds": {}}
    key = os.environ.get(plan.api_key_env)
    for sample in samples(plan):
        user_id, adds, source_ids = runner.prepare_adds(sample, plan.batch_size, plan.run_id)
        questions = sample.questions[: plan.max_questions]
        if all(question.question_id in indexed for question in questions):
            continue
        for payload in adds:
            ident = payload["request_id"]
            signature = digest(payload)
            if ident in state["adds"]:
                if state["adds"][ident]["digest"] != signature:
                    raise ValueError("Add checkpoint payload mismatch")
                continue
            started = time.perf_counter()
            response = runner.post_json(plan.base_url.rstrip("/") + "/add", payload, key, plan.timeout)
            if response.get("success") is not True or response.get("request_id") != ident:
                raise ValueError("Invalid Add response")
            state["adds"][ident] = {"digest": signature, "seconds": time.perf_counter() - started}
            atomic_json(checkpoint, state)
        for question in questions:
            if question.question_id in indexed:
                continue
            payload = {"user_id": user_id, "query": question.query, "top_k": plan.top_k}
            if question.options:
                payload["options"] = question.options
            started = time.perf_counter()
            hits = validate_hits(
                runner.post_json(plan.base_url.rstrip("/") + "/search", payload, key, plan.timeout), plan.top_k
            )
            record = {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "question_id": question.question_id,
                "user_id": user_id,
                "query": question.query,
                "answer": question.answer,
                "options": question.options,
                "evidence_refs": question.evidence_refs,
                "evidence_recall_at_k": runner.evidence_recall(
                    question, source_ids, {str(hit.get("id")) for hit in hits}
                ),
                "metadata": question.metadata,
                "results": hits,
                "search_seconds": time.perf_counter() - started,
            }
            append_row(path, record)
            indexed[question.question_id] = record


def pipeline_stage(
    pipeline: str,
    stage: str,
    inputs: Path,
    output: Path,
    expected: set[str],
    *,
    answers: Path | None = None,
    field: str | None = None,
    resume: bool = False,
) -> None:
    completed = index_rows(output, expected, repair_tail=resume)
    source = index_rows(inputs, expected)
    answer_rows = index_rows(answers, expected, repair_tail=resume) if answers else {}
    for ident, item in source.items():
        if ident in completed and not (stage == "evaluate" and judge_failed(completed[ident])):
            continue
        with tempfile.TemporaryDirectory(prefix="pipeline-", dir=output.parent) as temporary:
            work = Path(temporary)
            single_input, single_output = work / "input.jsonl", work / "output.jsonl"
            append_row(single_input, item)
            command = [pipeline, stage, "--input", str(single_input), "--output", str(single_output)]
            if answers:
                single_answer = work / "answer.jsonl"
                append_row(single_answer, answer_rows[ident])
                command.extend(["--answers", str(single_answer)])
            started = time.perf_counter()
            execute(command)
            rows = read_rows(single_output)
            if len(rows) != 1 or row_id(rows[0]) != ident:
                raise ValueError("Pipeline returned an unexpected question ID/count")
            row = rows[0]
            if stage == "answer":
                generated = row.get("generated_answer", row.get("model_output"))
                if not isinstance(generated, str) or not generated.strip():
                    raise ValueError("Pipeline returned no generated answer")
            if stage == "evaluate" and (
                not isinstance(row.get(field), (int, float))
                or not math.isfinite(row[field])
                or not 0 <= row[field] <= 1
            ):
                raise ValueError("Pipeline returned an invalid score")
            row["_benchmark_seconds"] = time.perf_counter() - started
            if ident in completed:
                completed[ident] = row
                # Replace a failed judge row without duplicating the question.
                replacement = output.with_suffix(output.suffix + ".replace")
                if replacement.exists():
                    replacement.unlink()
                for previous in completed.values():
                    append_row(replacement, previous)
                os.replace(replacement, output)
            else:
                append_row(output, row)
                completed[ident] = row


def report_details(directory: Path, name: str, result: dict, field: str) -> None:
    retrieval = read_rows(directory / f"{name}.retrieval.jsonl")
    expected = {str(row["question_id"]) for row in retrieval}
    answers = index_rows(directory / f"{name}.answers.jsonl", expected)
    scores = index_rows(directory / f"{name}.scores.jsonl", expected)
    search_times = [row["search_seconds"] for row in retrieval if "search_seconds" in row]
    result["retrieved_questions"] = len(retrieval)
    result["answered_questions"] = len(answers)
    result["completed_scores"] = len(scores)
    result["judge_failures"] = sum(judge_failed(row) for row in scores.values())
    result["scored_questions"] = sum(
        not judge_failed(row)
        and isinstance(row.get(field), (int, float))
        and math.isfinite(row[field])
        and 0 <= row[field] <= 1
        for row in scores.values()
    )
    result["pending_question_ids"] = (
        sorted(set(result.get("expected_ids", [])) - set(scores))
        if "score_field" in result or result.get("mode") == "all"
        else sorted(set(result.get("expected_ids", [])) - expected)
    )
    result["pending_questions"] = len(result["pending_question_ids"])
    result["perfect_score_questions"] = sum(not judge_failed(row) and row.get(field) == 1 for row in scores.values())
    result["judge_failure_question_ids"] = sorted(ident for ident, row in scores.items() if judge_failed(row))
    result["answer_seconds"] = sum(row.get("_benchmark_seconds", 0) for row in answers.values())
    result["judge_seconds"] = sum(row.get("_benchmark_seconds", 0) for row in scores.values())
    result["mean_search_seconds"] = sum(search_times) / len(search_times) if search_times else None
    result["p95_search_seconds"] = (
        sorted(search_times)[max(0, math.ceil(len(search_times) * 0.95) - 1)] if search_times else None
    )
    result["empty_retrieval_questions"] = sum(not row["results"] for row in retrieval)
    checkpoint = directory / f"{name}.checkpoint.json"
    adds = json.loads(checkpoint.read_text(encoding="utf-8"))["adds"] if checkpoint.exists() else {}
    result["completed_adds"] = len(adds)
    result["add_seconds"] = sum(row["seconds"] for row in adds.values())
    lines = [
        f"# {name} question details",
        "",
        "Full evidence and raw judge output are in the matching JSONL files.",
        "",
    ]
    for row in retrieval:
        ident = str(row["question_id"])
        answer = answers.get(ident, {})
        score = scores.get(ident, {})
        lines.extend(
            [
                f"## {ident}",
                "",
                f"Question: {row['query']}",
                "",
                f"Score: {score.get(field, 'pending')}; judge failure: {judge_failed(score)}",
                "",
                "Answer:",
                "",
                "    "
                + str(answer.get("generated_answer", answer.get("model_output", "pending"))).replace("\n", "\n    "),
                "",
            ]
        )
        reason = score.get("rubric_clbench_rationale", score.get("judge_response", ""))
        lines.extend(
            [
                f"Search seconds: {row.get('search_seconds', 'unknown')}; answer seconds: {answer.get('_benchmark_seconds', 'pending')}; judge seconds: {score.get('_benchmark_seconds', 'pending')}",
                "",
                "Judge explanation:",
                "",
                "    " + str(reason)[:1200].replace("\n", "\n    "),
                "",
                "Retrieved evidence (first three hits, up to 600 characters each):",
                "",
            ]
        )
        for hit in row["results"][:3]:
            lines.extend(
                [
                    f"Memory ID: {hit.get('id', 'not provided')}; score: {hit.get('score', 'not provided')}",
                    "",
                    "    " + hit["content"][:600].replace("\n", "\n    "),
                    "",
                ]
            )
    (directory / f"{name}.details.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(directory: Path, report: dict) -> None:
    atomic_json(directory / "summary.json", report)
    lines = [
        "# Public Memory benchmark",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "| Dataset | Status | Retrieved / target | Valid scores | Mean score | Judge failures | Time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, result in report["datasets"].items():
        score = result.get("mean_score")
        display = f"{score:.4f}" if score is not None else "—"
        lines.append(
            f"| [{name}]({name}.details.md) | {result['status']} | {result.get('retrieved_questions', 0)} / {result.get('target_questions', '—')} | {result.get('scored_questions', '—')} | {display} | {result.get('judge_failures', 0)} | {sum(value for phase, value in result.get('phase_seconds', {}).items() if phase != 'preflight'):.2f} |"
        )
        if result.get("failure_reason"):
            lines.extend(["", f"{name}, {result['phase']}: {result['failure_reason']}", ""])
    lines.extend(
        [
            "",
            "Scores follow different dataset criteria; this is not an official AML score.",
            "Mean score includes any judge-failure zeros; summary.json also reports mean_score_valid excluding those failures.",
            "Time includes pipeline process overhead and retries. Preflight timing is separate in summary.json.",
            "Each dataset has a details page and retrieval/input/answers/scores JSONL files.",
            "",
        ]
    )
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path | None:
    load_dotenv(args.env_file, override=False, interpolate=False)
    args._probe_cache = set()
    resume = getattr(args, "resume", False)
    check_only = getattr(args, "check_only", False)
    if args.questions < 1 or args.batch_size < 1 or args.top_k < 1 or args.timeout <= 0:
        raise ValueError("questions, batch-size, top-k and timeout must be positive")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("Choose each dataset only once")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, hyphens and underscores")
    args.base_url = validate_url(args.base_url)
    plans, signatures, sources = [], {}, {}
    for name in args.datasets:
        variant, pipeline, field = DATASETS[name]
        if name == "longmemeval":
            variant = getattr(args, "longmemeval_variant", "s")
        elif name == "beam":
            variant = getattr(args, "beam_variant", "100K")
        elif name == "personamem":
            variant = getattr(args, "personamem_variant", "32k")
        single = name in {"longmemeval", "personamem", "clbench-life"}
        if name == "scriptmem":
            variant = None if args.scriptmem_source == "all" else args.scriptmem_source
            limit = len(runner.SCRIPTMEM_SOURCES) if variant is None else 1
        else:
            limit = args.questions if single else 1
        plan = argparse.Namespace(
            dataset=name,
            variant=variant,
            limit=limit,
            max_questions=1 if single else args.questions,
            batch_size=args.batch_size,
            top_k=args.top_k,
            run_id=args.run_id,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            scriptmem_conversations_path=getattr(
                args, "scriptmem_conversations", runner.SCRIPTMEM_CONVERSATIONS
            ),
            scriptmem_qa_types=args.scriptmem_qa_types if name == "scriptmem" else None,
        )
        signatures[name] = inspect_plan(plan)
        sources[name] = source_provenance(name, plan)
        plans.append((name, plan, pipeline, field))
    if args.dry_run:
        print("Dry run completed: no Memory or model API calls, no output files.")
        return None
    if any(name == "scriptmem" and signatures[name]["adds"] == 0 for name in args.datasets):
        raise ValueError("ScriptMem has no Add records; check the restored conversations-only JSONL input.")
    required = [] if args.allow_unauthenticated else [args.api_key_env]
    if args.stage == "all":
        required.extend(["ANSWER_API_BASE", "ANSWER_API_KEY", "ANSWER_MODEL"])
        if any(name not in {"personamem", "scriptmem"} for name in args.datasets):
            required.extend(["JUDGE_API_BASE", "JUDGE_API_KEY", "JUDGE_MODEL"])
    missing = [key for key in required if not os.environ.get(key, "").strip()]
    if missing:
        raise ValueError("Missing environment variables: " + ", ".join(missing))
    for key in required:
        if key.endswith("_API_BASE"):
            validate_url(os.environ[key])
    if check_only:
        preflight(
            args, memory=True, answer=args.stage == "all", judge=args.stage == "all" and "JUDGE_API_BASE" in required
        )
        print("Preflight passed. No dataset histories imported and no output directory created.")
        return None
    directory = args.output_dir.resolve() / args.run_id
    if resume:
        if not (directory / "manifest.json").is_file():
            raise ValueError("--resume requires an existing run with manifest.json")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory):
        return saved_run(args, directory, plans, signatures, sources)


def saved_run(args, directory: Path, plans: list, signatures: dict, sources: dict) -> Path:
    resume = getattr(args, "resume", False)
    manifest = {
        "version": 1,
        "base_url": args.base_url,
        "questions": args.questions,
        "scriptmem_qa_types": args.scriptmem_qa_types if "scriptmem" in args.datasets else None,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "datasets": signatures,
        "dataset_sources": sources,
        "code_sha256": digest(
            {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in [
                    "text_memory_runner.py",
                    "prepare_pipeline_inputs.py",
                    "api_config.py",
                    "run_benchmark.py",
                    "benchmark_state.py",
                    *[pipeline for _, _, pipeline, _ in plans],
                ]
            }
        ),
    }
    generation = {
        key: os.environ.get(key, "") for key in ("ANSWER_API_BASE", "ANSWER_MODEL", "JUDGE_API_BASE", "JUDGE_MODEL")
    }
    if resume:
        previous = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if {key: value for key, value in previous.items() if key != "generation"} != manifest:
            raise ValueError("Resume configuration/data/code mismatch; use the original settings or start a new run")
        if args.stage == "all" and previous.get("generation", generation) != generation:
            raise ValueError("Resume generation model/endpoint mismatch")
        manifest = previous
    if args.stage == "all":
        manifest["generation"] = generation
    atomic_json(directory / "manifest.json", manifest)
    report = (
        json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if resume and (directory / "summary.json").exists()
        else {"run_id": args.run_id, "datasets": {}}
    )
    report.update(
        stage=args.stage,
        requested_questions_per_dataset=args.questions,
        top_k=args.top_k,
        batch_size=args.batch_size,
        answer_model=generation["ANSWER_MODEL"],
        judge_model=generation["JUDGE_MODEL"],
    )
    failures = False
    for name, plan, pipeline, field in plans:
        expected = set(signatures[name]["question_ids"])
        retrieval, normalized, answers, scores = [
            directory / f"{name}.{suffix}.jsonl" for suffix in ("retrieval", "input", "answers", "scores")
        ]
        result = report["datasets"].setdefault(name, {})
        result.update(
            status="running",
            phase="preflight",
            target_questions=len(expected),
            mode=args.stage,
            expected_ids=sorted(expected),
        )
        for key in ("error_type", "failure_reason"):
            result.pop(key, None)
        result.setdefault("phase_seconds", {})
        phase_start = time.perf_counter()
        try:
            done_retrieval = index_rows(retrieval, expected, repair_tail=resume)
            done_answers = index_rows(answers, expected, repair_tail=resume)
            done_scores = index_rows(scores, expected, repair_tail=resume)
            need_answer = args.stage == "all" and set(done_answers) != expected
            need_judge = args.stage == "all" and (
                set(done_scores) != expected or any(judge_failed(row) for row in done_scores.values())
            )
            try:
                preflight(
                    args,
                    memory=set(done_retrieval) != expected,
                    answer=need_answer,
                    judge=need_judge and name not in {"personamem", "scriptmem"},
                )
            finally:
                result["phase_seconds"]["preflight"] = (
                    result["phase_seconds"].get("preflight", 0) + time.perf_counter() - phase_start
                )
            for phase in (
                ["retrieval"] if args.stage == "retrieval" else ["retrieval", "conversion", "answer", "evaluate"]
            ):
                result["phase"] = phase
                phase_start = time.perf_counter()
                write_report(directory, report)
                try:
                    if phase == "retrieval":
                        run_retrieval(plan, retrieval, directory / f"{name}.checkpoint.json", expected, resume=resume)
                        if set(index_rows(retrieval, expected)) != expected:
                            raise ValueError("Incomplete retrieval question set")
                    elif phase == "conversion":
                        rows = [convert_record(row) for row in read_rows(retrieval)]
                        if normalized.exists():
                            if read_rows(normalized) != rows:
                                raise ValueError("Normalized input changed")
                        else:
                            replacement = normalized.with_suffix(".tmp")
                            if replacement.exists():
                                replacement.unlink()
                            for row in rows:
                                append_row(replacement, row)
                            os.replace(replacement, normalized)
                    else:
                        pipeline_stage(
                            pipeline,
                            phase,
                            normalized,
                            answers if phase == "answer" else scores,
                            expected,
                            answers=answers if phase == "evaluate" else None,
                            field=field,
                            resume=resume,
                        )
                finally:
                    result["phase_seconds"][phase] = (
                        result["phase_seconds"].get(phase, 0) + time.perf_counter() - phase_start
                    )
            if args.stage == "all":
                result.update(summarize(retrieval, scores, field))
            result.update(status="complete", phase="complete")
            if args.stage == "all" and result.get("judge_failures"):
                failures = True
                result.update(
                    status="judge_failed",
                    phase="evaluate",
                    failure_reason="Judge failures were saved as zero scores; resume to retry only failed judges",
                )
        except Exception as error:
            failures = True
            result.update(status="failed", error_type=type(error).__name__, failure_reason=safe_failure(error))
            print(f"{name}: {result['phase']} failed ({type(error).__name__}).", file=sys.stderr)
        except KeyboardInterrupt:
            result.update(status="interrupted", failure_reason="Interrupted; resume this run to continue")
            raise
        finally:
            try:
                report_details(directory, name, result, field)
            except ValueError:
                result["details_error"] = "Invalid or incomplete stage file; inspect JSONL before resuming"
            write_report(directory, report)
    print(f"Report: {directory / 'REPORT.md'}")
    if failures:
        raise SystemExit(1)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=DEFAULT_DATASETS)
    parser.add_argument("--questions", type=int, default=10, help="Target questions per dataset")
    parser.add_argument("--longmemeval-variant", choices=["s", "m", "oracle"], default="s")
    parser.add_argument("--personamem-variant", choices=["32k"], default="32k")
    parser.add_argument("--beam-variant", choices=["100K", "500K", "1M"], default="100K")
    parser.add_argument(
        "--scriptmem-source",
        choices=["all", *runner.SCRIPTMEM_SOURCES],
        default="all",
        help="ScriptMem source subset; --questions caps the total question count",
    )
    parser.add_argument(
        "--scriptmem-qa-types",
        nargs="+",
        choices=["single_choice", "multi_select", "ordering"],
        help="Select ScriptMem answer formats round-robin (useful for small smoke runs)",
    )
    parser.add_argument(
        "--scriptmem-conversations",
        type=Path,
        default=runner.SCRIPTMEM_CONVERSATIONS,
        help="Add-only ScriptMem conversations JSONL (no QA fields)",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8899")
    parser.add_argument("--stage", choices=["all", "retrieval"], default="all")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--api-key-env", default="CHALLENGE_API_KEY")
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Probe Memory/model APIs without importing histories; model probes may be billed",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Continue an existing --run-id, skipping durable completed work"
    )
    args = parser.parse_args()
    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id")
    args.run_id = args.run_id or "eval-" + uuid.uuid4().hex
    print(f"Run ID: {args.run_id}")
    try:
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Configuration/input error: {error}\n")


if __name__ == "__main__":
    main()
