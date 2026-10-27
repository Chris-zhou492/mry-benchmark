# Dataset inventory

This release is a source-pinned data pack, not a frozen experimental split. The unified runner integrates LoCoMo, LongMemEval S/M/Oracle, PersonaMem v1 32K, CL-bench Life, and BEAM 100K/500K/1M with dataset-specific answer/scoring pipelines. ScriptMem QA and its exact option evaluator are included, but full Add/Search evaluation needs source dialogue that the official release omits. PersonaMem v2 data can be retrieved, but its loader is not integrated yet. Raw scores across these datasets are not interchangeable and must not be pooled into one accuracy figure; see [RUN_BENCHMARK.md](RUN_BENCHMARK.md).

## Included files

| Dataset key | Bundled files | Upstream version | License / limitation |
| --- | --- | --- | --- |
| `locomo` | `datasets/bundled/locomo/locomo10.json` | Snap Research LoCoMo commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | CC BY-NC 4.0; text-only use. Image assets are not included. |
| `scriptmem` | `questions.jsonl`, `conversations.jsonl` | Memorax ScriptMem commit `831ca3e3123dbaed53fdf7d80fde066bed9f5abc` | CC BY-NC 4.0 for benchmark-created materials. The public conversations file is an omission notice/schema example, not source dialogue. |
| `personamem-v1-32k` | 32K questions CSV and shared contexts JSONL | HF revision `a8076d5608c93ba2a28983cd78aa99b01a163ae7` | MIT. |
| `personamem-v2-text` | Text benchmark CSV | HF revision `ed956dea41521fc4499acbc63f966e0fd3c053ba` | CC BY 4.0. Download also retrieves its 981 linked history files (about 0.5 GB) from the same pinned revision. |
| `cl-bench-life` | `CL-bench Life.jsonl` and upstream `LICENSE.txt` | HF revision `cd2589fab8ed0af47b2c96b4e104478d72560ebe` | Evaluation/testing/benchmarking only. No training, fine-tuning, calibration, distillation, adaptation, or parameter updates. |
| `beam-100k` | 100K parquet | HF revision `3205395e897e7318c7b094ef4e6047b9b82dbb03` | CC BY-SA 4.0. |
| `beam-500k` | 500K parquet | Same pinned BEAM revision | CC BY-SA 4.0. |
| `beam-1m` | 1M parquet | Same pinned BEAM revision | CC BY-SA 4.0. |
| `longmemeval-oracle` | Oracle JSON | HF revision `98d7416c24c778c2fee6e6f3006e7a073259d48f` | MIT metadata. |

## Official-source downloads

These files are not stored in this Git repository. The downloader fetches them from pinned official revisions and verifies the advertised size and content hash when the upstream exposes one.

| Dataset key | Files | Approximate size |
| --- | --- | ---: |
| `longmemeval-s` | Cleaned S JSON | 277 MB |
| `longmemeval-m` | Cleaned M JSON | 2.74 GB |
| `beam-10m` | Two parquet shards | 344 MB total |

PersonaMem v2's history files are included in `personamem-v2-text` retrieval. ScriptMem's public release has 457 question records over four scripted works, but no original scripts or full conversations; use it only for evaluator/scorer-path checks unless rights-cleared source text is obtained separately. It is not an end-to-end Add/Search memory benchmark in this release.

## Retrieve and verify

```powershell
python tools/manage_datasets.py list
python tools/manage_datasets.py download longmemeval-s
python tools/manage_datasets.py download beam-10m
python tools/manage_datasets.py download personamem-v2-text
python tools/manage_datasets.py verify
```

The script is standard-library-only and writes downloaded files under `datasets/downloads/`. It pins source revisions; it does not install or execute upstream code. Avoid downloading `longmemeval-m` unless the local machine has sufficient disk space.

## Sources and papers

- LongMemEval: [paper](https://arxiv.org/abs/2410.10813), [official repository](https://github.com/xiaowu0162/LongMemEval), [official dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned).
- LoCoMo: [ACL paper](https://aclanthology.org/2024.acl-long.747/), [official repository](https://github.com/snap-research/LoCoMo).
- PersonaMem v1: [paper](https://arxiv.org/abs/2504.14225), [official repository](https://github.com/bowen-upenn/PersonaMem), [official dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1).
- PersonaMem v2: [paper](https://arxiv.org/abs/2512.06688), [official repository](https://github.com/bowen-upenn/PersonaMem-v2), [official dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2).
- CL-bench Life: [project](https://www.clbench.com/), [official repository](https://github.com/Tencent-Hunyuan/CL-bench), [official dataset and terms](https://huggingface.co/datasets/tencent/CL-bench-Life).
- BEAM: [paper](https://arxiv.org/abs/2510.27246), [official repository](https://github.com/mohammadtavakoli78/BEAM), [BEAM data](https://huggingface.co/datasets/Mohammadta/BEAM), [BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M).
- ScriptMem: [official repository](https://github.com/memorax-ai/ScriptMem), [data notes](https://github.com/memorax-ai/ScriptMem/blob/main/data/README.md).
