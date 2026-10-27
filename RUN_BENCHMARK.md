# Unified benchmark runner

`run_benchmark.py` is the primary test entry point. It runs the same Add/Search contract for each selected dataset, then optionally passes returned evidence through a fixed answer and evaluation pipeline. The answer and judge services are configured independently of the Memory service.

## 1. Prepare the environment

From the repository root, install Python 3.12 and `uv`, then create the Hatch evaluation environment:

```powershell
uv python install 3.12
uvx --from hatch hatch run eval-fastapi:python --version
```

For a new checkout, copy the credential template and fill it in:

```powershell
Copy-Item .env.example .env
```

Set `CHALLENGE_API_KEY` to protect your Memory adapter. Full answer-and-score runs also need `ANSWER_API_BASE`, `ANSWER_API_KEY`, and `ANSWER_MODEL`. Dataset-specific LLM judges need `JUDGE_API_BASE`, `JUDGE_API_KEY`, and `JUDGE_MODEL`. PersonaMem v1 and ScriptMem use exact local option scoring and do not need a judge endpoint. Model endpoints must provide an OpenAI-compatible `/chat/completions` API.

The runner reads `.env` automatically. Process environment variables take precedence; use `--env-file path/to/file` to choose another file. Do not commit `.env` or put credentials in `--base-url`.

## 2. Fetch data needed for your run

LoCoMo, PersonaMem v1 32K, CL-bench Life, BEAM 100K/500K/1M, ScriptMem QA, and LongMemEval Oracle are bundled. LongMemEval S/M are fetched from pinned official revisions:

```powershell
python tools/manage_datasets.py list
python tools/manage_datasets.py download longmemeval-s
python tools/manage_datasets.py verify
```

LongMemEval M is about 2.74 GB. BEAM Parquet support is included in the `eval-fastapi` Hatch environment. Data terms vary; read [DATASET_LICENSES.md](DATASET_LICENSES.md) before using or redistributing each dataset.

## 3. Start or connect a Memory implementation

The adapter contract is documented in [protocols/memory-http.md](protocols/memory-http.md). If your method already exposes compatible `POST /add` and `POST /search`, use its address. Otherwise, adapt the provided server by implementing the synchronous methods below in a class and starting it with `--backend module:Class`:

```python
class MyMemory:
    def add(self, *, request_id, user_id, session_id, messages):
        # Write messages to the memory method and make request_id idempotent.
        ...

    def search(self, *, user_id, query, top_k, options=None):
        # Return a list of evidence hits with id, content, and optional score.
        ...
```

The included `DemoMemory` is a local keyword-matching example stored in RAM. It is useful for checking the request flow, but is not a meaningful research baseline. The adapter's default authentication uses `CHALLENGE_API_KEY` from `.env`.

```powershell
uvx --from hatch hatch run eval-fastapi:python memory_adapter.py --port 8899
```

## 4. Check data and run Add/Search

Use dry-run to inspect the selected samples without network calls or result files:

```powershell
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py --datasets locomo --questions 2 --dry-run
```

Then run retrieval only. This stage does not require answer or judge model credentials:

```powershell
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py `
  --base-url http://127.0.0.1:8899 `
  --datasets locomo `
  --questions 10 `
  --stage retrieval
```

`--questions` is the target number per dataset. It caps question count for a dataset sample and selects the earliest records in source order; it does not randomize. LoCoMo and BEAM use one conversation with up to N questions. LongMemEval, PersonaMem, and CL-bench use N single-question histories. ScriptMem counts questions across selected works.

Available datasets are `longmemeval`, `locomo`, `personamem`, `clbench-life`, `beam`, and `scriptmem`. The default is `locomo`. Additional options are:

- `--longmemeval-variant s|m|oracle` (default `s`); Oracle is diagnostic and must be reported separately.
- `--personamem-variant 32k` (currently the integrated v1 variant).
- `--beam-variant 100K|500K|1M`.
- `--scriptmem-source all|angry|enemy|friends|man_earth` and `--scriptmem-qa-types single_choice multi_select ordering`.
- `--batch-size`, `--top-k`, `--timeout`, and `--output-dir` to set run parameters.

## 5. Run complete answer and scoring stages

After setting reader/judge credentials and downloading the required dataset files:

```powershell
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py `
  --base-url http://127.0.0.1:8899 `
  --datasets locomo longmemeval personamem clbench-life beam `
  --questions 10 `
  --stage all
```

The default full-run setting is LoCoMo, so select each additional dataset explicitly. Keep a fixed answer model, judge model, `top_k`, and dataset revision when comparing memory methods. Dataset-native criteria differ; do not average them together as one cross-dataset accuracy.

### ScriptMem data limitation

The public ScriptMem package contains 457 questions and scoring labels, but does not contain the original scripts or full conversations. The bundled `conversations.jsonl` is only a format/omission notice. Run `python tools/scriptmem_smoke.py` or `--dry-run` to inspect the QA set; these commands do not test memory retrieval. A real Add/Search run requires a separate `--scriptmem-conversations` JSONL file with one source-cleared conversation per source/sample. The repository does not distribute the community-shared third-party scripts. Any result using separately obtained conversation text must document its source and completeness limitation; it is not an official full reproduction.

## Preflight, outputs, and resume

`--check-only` performs API readiness probes without importing benchmark histories or creating a result directory. It makes small real model requests when the selected stage needs those models; provider charges may apply. `--dry-run` is offline and never makes model/API requests.

Each run writes to `outputs/<run-id>/`:

- `REPORT.md` and `summary.json` with dataset status, score fields, counts, judge failures, and phase timing.
- Per-question `<dataset>.details.md` summaries.
- `<dataset>.retrieval.jsonl`, normalized inputs, generated answers, and scores.
- A code/data manifest and per-stage checkpoints for reproducibility.

Use `--run-id my-run` to select a run directory. If interrupted, repeat the same configuration with `--resume`. The runner checks the data, code, endpoint, question selection, and generation model settings against the saved manifest, then skips work already written. One RunId cannot run concurrently. If an Add response is lost, the adapter must make the same `request_id` idempotent.

The pipeline can report judge failures separately from system mistakes; see `summary.json`. Runs are research artifacts, not official leaderboard submissions. Results depend on the selected Memory implementation and external answer/judge model configuration.
