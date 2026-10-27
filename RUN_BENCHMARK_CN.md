# 统一 benchmark 使用说明

`run_benchmark.py` 是统一入口：按数据集加载历史记录，将历史分批写入被测 Memory 的 `/add`，逐题调用 `/search`，再运行对应的答案生成和评分 pipeline。Memory 服务只需实现文档规定的 HTTP 接口；答案生成器和裁判模型由评测运行者单独配置。

## 快速开始

需要 Python 3.12 和 `uv`。在仓库根目录运行：

```powershell
uv python install 3.12
uvx --from hatch hatch run eval-fastapi:python --version
Copy-Item .env.example .env
```

编辑 `.env`：设置 `CHALLENGE_API_KEY`。完整答案评测还需要 `ANSWER_API_BASE`、`ANSWER_API_KEY`、`ANSWER_MODEL`；LongMemEval、LoCoMo、CL-bench 和 BEAM 还需要 `JUDGE_API_BASE`、`JUDGE_API_KEY`、`JUDGE_MODEL`。PersonaMem v1 和 ScriptMem 使用本地确定性选项评分，不需要裁判模型。API 地址应兼容 OpenAI `/chat/completions` 协议。

LoCoMo、PersonaMem v1 32K、CL-bench Life、BEAM 100K/500K/1M、ScriptMem QA 和 LongMemEval Oracle 已随仓库提供。LongMemEval S/M 先从固定的官方版本下载：

```powershell
python tools/manage_datasets.py download longmemeval-s
python tools/manage_datasets.py verify
```

长上下文 BEAM 读取需要 `pyarrow`，已包含在 `eval-fastapi` 环境。使用前查看 [数据集和许可说明](DATASET_LICENSES.md)。

## 接入 Memory 服务

已有 `/add`、`/search` 服务可直接指定 `--base-url`。没有 HTTP 接口时，可将自己的 SDK 包装到 `memory_adapter.py` 的 `add()` 和 `search()` 方法里。仓库自带的 `DemoMemory` 仅用于连通性检查，内存存储、英文关键词匹配，不代表研究基线。

```powershell
# 终端一：启动演示 API
uvx --from hatch hatch run eval-fastapi:python memory_adapter.py --port 8899

# 终端二：离线检查题目与数据加载，不发 API 请求
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py --datasets locomo --questions 2 --dry-run

# 只测 Add/Search，不需要答案模型和裁判模型
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py `
  --base-url http://127.0.0.1:8899 `
  --datasets locomo `
  --questions 10 `
  --stage retrieval
```

要跑完整答案生成和评分，先在 `.env` 填好模型配置，再将 `--stage retrieval` 改为 `--stage all`。例：

```powershell
uvx --from hatch hatch run eval-fastapi:python run_benchmark.py `
  --base-url http://127.0.0.1:8899 `
  --datasets locomo longmemeval personamem clbench-life beam `
  --questions 10 `
  --stage all
```

默认只跑 LoCoMo。选择额外数据集时使用 `--datasets`；每组目标题数由 `--questions` 设置。LongMemEval 默认是 S 版，可用 `--longmemeval-variant s|m|oracle` 选择；Oracle 仅作诊断，单独报告。BEAM 可用 `--beam-variant 100K|500K|1M` 选择。当前 PersonaMem 集成版本为 v1 32K。

`--check-only` 会真实探测服务，但不导入历史；模型探测会产生少量 API 调用。`--dry-run` 完全离线。不要把 API token 写进 URL 或提交 `.env`。

## ScriptMem 限制

仓库包含官方发布的 457 道 ScriptMem 题目及答案选项，不包含原始剧本/对话。`datasets/bundled/scriptmem/conversations.jsonl` 是缺失说明和 schema 示例，不是测试历史。因此可以用 `--dry-run` 或 `python tools/scriptmem_smoke.py` 检查题目格式，但不能据此进行 Add/Search 或报告记忆成绩。

如果另行取得获准使用的对话，提供只包含原始对话的 JSONL：

```powershell
python run_benchmark.py --datasets scriptmem --questions 10 --scriptmem-conversations path/to/conversations-only.jsonl --stage retrieval
```

官方 QA 的许可不覆盖第三方剧本。请记录对话来源和完整性限制；使用社区补充文本的结果不能称为官方完整复现。

## 输出和恢复

结果写到 `outputs/<run-id>/`，包括 `REPORT.md`、`summary.json`、逐题详情、检索结果、答案、评分和阶段检查点。用 `--run-id my-run` 固定实验 ID；中断后原参数追加 `--resume`，runner 会检查数据、关键代码、endpoint、模型和题目选择，并跳过已完成阶段。不要对同一 RunId 并发启动两个进程。

不同数据集的任务和评分标准不同，不要把它们的均分合并为一个准确率。结果是本项目的独立研究评测，不是 AML 官方成绩。完整参数与接口格式见 [英文运行手册](RUN_BENCHMARK.md) 和 [Memory API 协议](protocols/memory-http.md)。
