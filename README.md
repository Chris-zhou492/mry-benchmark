# MRY-Benchmark

MRY-Benchmark is a public research workspace for long-term memory evaluation. This first release contains the independent pipeline prototype, synthetic fixtures, selected benchmark files, and tools for retrieving pinned upstream data.

**Research status:** this is not a completed benchmark suite. Dataset adapters, a production reader/scorer, protocol freeze, and formal comparative results are still in progress. The included mock pipeline is offline engineering validation; its scores are not benchmark results.

## Contents

- `src/`, `protocols/`, `tests/`, and `tools/`: the independent Python pipeline prototype and synthetic/mock validation tools.
- `datasets/bundled/`: selected, version-pinned upstream benchmark files that fit ordinary GitHub file limits.
- `datasets/manifest.json` and `tools/manage_datasets.py`: dataset provenance, download, and integrity checks.
- `DATASETS.md`: dataset scope, pinned versions, license notes, and usage.
- `DATASET_LICENSES.md`: attribution and redistribution conditions for each included dataset.

Large files are fetched from their official upstream releases on demand. No model or Memory API call is made by the dataset manager.

## Quick start

Requires Python 3.12. From the repository root:

```powershell
python tools/manage_datasets.py list
python tools/manage_datasets.py verify
python tools/scriptmem_smoke.py
python tools/manage_datasets.py download longmemeval-s
python tools/manage_datasets.py download personamem-v2-text
```

`download` accepts one dataset key at a time. Large downloads are written under `datasets/downloads/`, which is ignored by Git. See [DATASETS.md](DATASETS.md) before selecting data for an experiment.

The pipeline prototype can be run with Hatch 1.18.1:

```powershell
uvx --from 'hatch==1.18.1' hatch run test
uvx --from 'hatch==1.18.1' hatch run mock-http
```

The ScriptMem command checks its 457 public QA rows and reports whether source dialogues are present. These commands validate data integrity and synthetic/mock flows only; they do not run a benchmark or produce system-performance claims.

## Data and code terms

Each dataset has its own upstream license and attribution requirements. Read [DATASET_LICENSES.md](DATASET_LICENSES.md) before use or redistribution. ScriptMem's published package contains questions but omits its source scripts and conversations, so it cannot support end-to-end memory ingestion evaluation.

The repository's original code and documentation do not yet have a separately selected license. Dataset licenses do not grant rights to that code or documentation.

## Citation and changes

The upstream papers and source revisions are listed in [DATASETS.md](DATASETS.md). Cite the original dataset and paper when using any upstream data. MRY-Benchmark does not claim ownership of third-party datasets.
