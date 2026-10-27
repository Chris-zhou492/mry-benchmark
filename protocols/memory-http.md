# MRY Memory API adapter v0.1

The unified runner calls a system under test over HTTP. A thin wrapper can map these requests to any memory SDK, service, or database.

## Authentication

The runner sends `X-Api-Key: <value>` using the environment variable named by `--api-key-env` (default `CHALLENGE_API_KEY`). The key is not included in result manifests. Base URLs must be HTTP(S) URLs without credentials, query parameters, or fragments.

## `POST /add`

```json
{
  "request_id": "public-eval:run-id:locomo:conv-26:D1:0",
  "user_id": "public-eval:run-id:locomo:conv-26",
  "session_id": "public-eval:run-id:locomo:conv-26:session_1",
  "messages": [
    {"role": "user", "content": "[source_date: ...] [speaker] ..."},
    {"role": "assistant", "content": "...", "timestamp": 1704067200000}
  ]
}
```

Roles are `user` or `assistant`; timestamps are optional Unix milliseconds. Add batches are grouped by source session and bounded by `--batch-size`. Repeating the same `request_id` and payload must not duplicate memories; reusing an ID with a different payload should fail. The write must be searchable when `/add` returns.

Successful response:

```json
{"success": true, "request_id": "public-eval:run-id:locomo:conv-26:D1:0"}
```

## `POST /search`

```json
{
  "user_id": "public-eval:run-id:locomo:conv-26",
  "query": "What happened at the support group?",
  "top_k": 10,
  "options": ["option A", "option B"]
}
```

`options` appears for multiple-choice questions and may be ignored by the memory method. Return at most `top_k` hits in descending relevance order:

```json
{
  "data": [
    {"id": "memory-123", "content": "A source memory with useful evidence.", "score": 0.91}
  ]
}
```

Each hit must be an object with nonempty string `content`. A stable `id` and numeric `score` are recommended; original IDs let the runner calculate evidence recall when a dataset supplies source-message labels. Empty retrieval is `{"data": []}`.

## Scope and safety

Each dataset sample receives a unique `user_id`; different samples and runs must not share search results. Questions and multiple-choice options may be sent to `/search`; gold answers and scoring rubrics are kept out of `/add` and `/search` and are used only by the local downstream answer/evaluation stages. Preserve message roles where possible; ScriptMem's fictional dialogue is intentionally sent as assistant observations by its dataset adapter.

`memory_adapter.py` provides a FastAPI implementation with authentication, request validation, and in-process duplicate Add protection. Persistent backends remain responsible for idempotency across restarts and for user isolation.
