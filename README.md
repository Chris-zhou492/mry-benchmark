# MRY-Benchmark

MRY-Benchmark is an open, end-to-end evaluation suite for long-term memory systems. One runner handles history ingestion, retrieval, answer generation, task-specific scoring, checkpointing, and reports across multiple public datasets. A tested memory system only needs to provide the documented `/add` and `/search` API; the included FastAPI adapter is a disposable example backend.

**Research status:** this is an independent research benchmark, not an official AML submission or leaderboard score. Dataset criteria differ and must be reported separately. LongMemEval S/M, LoCoMo, PersonaMem v1, CL-bench Life, and BEAM are integrated. ScriptMem's 457 public QA records are included, but the official release omits source conversations, so full memory evaluation requires separately obtained, rights-cleared conversation text. PersonaMem v2 is not integrated yet.

## Quick start

Requires Python 3.12 and `uv`. From this repository directory:

```powershell
# Install the benchmark, model-pipeline, BEAM, and example-adapter dependencies.
uvx --from hatch hatch run eval-fastapi:python --version
Copy-Item .env.example .env

# Inspect two LoCoMo questions without API calls or output files.
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py --datasets locomo --questions 2 --dry-run

# Start the example memory adapter in one terminal. It requires CHALLENGE_API_KEY from .env.
uvx --from hatch hatch run eval-fastapi:python memory_adapter.py --port 8899

# In another terminal, test Add/Search only. No answer/judge model keys are needed.
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py --base-url http://127.0.0.1:8899 --datasets locomo --questions 2 --stage retrieval
```

The demo backend does keyword matching in RAM. Replace its `DemoMemory` class with your own storage and retrieval implementation before interpreting scores. For complete answer generation and evaluation, configure the separate reader and judge models in `.env`, then run without `--stage retrieval`.

## Integrated datasets

| Dataset | Runner support | Data availability |
| --- | --- | --- |
| LoCoMo | Add/Search, answer generation, binary judge | Bundled, text-only. |
| LongMemEval S/M | Add/Search, answer generation, binary judge | Download pinned S or M files with `tools/manage_datasets.py`; Oracle is available for diagnostics. |
| PersonaMem v1 32K | Add/Search, answer generation, exact option scoring | Bundled. The hidden system persona is not sent as memory. |
| CL-bench Life | Add/Search, answer generation, rubric judge | Bundled. Evaluation-only terms apply. |
| BEAM 100K/500K/1M | Add/Search, answer generation, rubric judge | Bundled; requires the optional `pyarrow` dependency, included in the `eval-fastapi` environment. |
| ScriptMem | QA parsing and exact option evaluator; Add/Search only when source dialogue is supplied | Official questions are bundled. The official release does not include original scripts/conversations. The bundled conversation file is an omission notice, not test history. |

See the [中文使用说明](RUN_BENCHMARK_CN.md) or [English guide](RUN_BENCHMARK.md) for dataset downloads, API configuration, the comprehensive run, result artifacts, resume behavior, and metric interpretation. See [DATASETS.md](DATASETS.md) and [DATASET_LICENSES.md](DATASET_LICENSES.md) before using benchmark data.

## Contents

- `run_benchmark.py`: unified preflight, retrieval, answer, scoring, checkpoint, and report entry point.
- `text_memory_runner.py`, `prepare_pipeline_inputs.py`, `benchmark_state.py`: dataset loading, Add/Search adapter, data conversion, and durable run state.
- `data/`: task-specific answer and scoring pipelines for each integrated benchmark.
- `memory_adapter.py`: replaceable FastAPI adapter and non-persistent demo backend.
- `src/memory_benchmark/`: independent artifact and evidence-pipeline prototype.
- `tools/manage_datasets.py`: pinned dataset retrieval and checksum verification.
- `datasets/manifest.json`: dataset versions, official-source URLs, SHA-256 hashes, and bundled-file mappings.

## Reporting

The full runner writes `REPORT.md`, `summary.json`, per-question details, normalized model inputs, generated answers, scores, and retrieval records under `outputs/<run-id>/`. Checkpoints let `--resume` continue an interrupted run without repeating completed Add, Search, answer, and valid scoring calls. Credentials are loaded from `.env` or the process environment and are not written to run manifests.

An end-to-end result depends on the selected Memory implementation, answer model, judge model, data release, and run configuration. Use the saved manifest and dataset citations when reporting results. The repository does not claim ownership of third-party datasets or official benchmark scores.
