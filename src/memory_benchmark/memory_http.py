"""Independent stdlib HTTP client for public Memory-method adapters."""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    AddReceipt,
    HistoryArtifact,
    PublicQuestionArtifact,
    RetrievalArtifact,
    RetrievalRequest,
    SourceMessageArtifact,
    Timing,
    Usage,
)
from .schemas import SchemaError, canonical, digest, loads, require, sha, text


class MemoryHttpError(RuntimeError):
    """Safe-to-log HTTP failure; provider response bodies are intentionally omitted."""

    def __init__(self, code: str, attempts: int):
        self.code = code
        self.attempts = attempts
        super().__init__(f"Memory HTTP operation failed: {code} after {attempts} attempt(s)")


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    backoff_seconds: float = 0.0
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)

    def __post_init__(self):
        require(type(self.max_attempts) is int and 1 <= self.max_attempts <= 10, "max_attempts must be 1..10")
        require(type(self.backoff_seconds) in (int, float) and self.backoff_seconds >= 0, "Invalid backoff")
        require(
            type(self.retry_statuses) is tuple
            and all(type(code) is int and 100 <= code <= 599 for code in self.retry_statuses),
            "Invalid retry status list",
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_json(raw: bytes) -> object:
    try:
        return loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, SchemaError, ValueError) as error:
        raise ProtocolError("Server returned invalid JSON") from error


def namespace_for(run_id: str, method_id: str, question_id: str, prefix_sha256: str) -> str:
    text(run_id, "run_id")
    text(method_id, "method_id")
    text(question_id, "question_id")
    sha(prefix_sha256, "prefix_sha256")
    seed = digest(
        {"run_id": run_id, "method_id": method_id, "question_id": question_id, "prefix_sha256": prefix_sha256}
    )
    return "mb-" + seed[:40]


def validated_public_prefix(
    history: HistoryArtifact, question: PublicQuestionArtifact
) -> tuple[SourceMessageArtifact, ...]:
    require(
        type(history) is HistoryArtifact and type(question) is PublicQuestionArtifact,
        "Expected typed history and question",
    )
    require(
        (history.history_id, history.family_id, history.dataset_id)
        == (question.history_id, question.family_id, question.dataset_id),
        "History identity mismatch",
    )
    matches = [
        index
        for index, item in enumerate(history.messages)
        if item.message_id == question.allowed_history.last_message_id
    ]
    require(bool(matches), "Unknown allowed-history cutoff")
    cutoff = matches[0]
    require(
        history.messages[cutoff].source_order == question.allowed_history.last_source_order,
        "Allowed-history order mismatch",
    )
    prefix = history.messages[: cutoff + 1]
    require(
        not any(item.visibility == "unclassified_system" for item in prefix), "Unclassified system message in prefix"
    )
    public = tuple(item for item in prefix if item.visibility == "public_history")
    require(
        digest([item.to_dict() for item in public]) == question.allowed_history.prefix_sha256,
        "Allowed-history hash mismatch",
    )
    return public


def _transport_message(item: SourceMessageArtifact) -> dict[str, str]:
    if item.speaker_type == "human":
        role = "user"
        content = f"[speaker_id:{item.speaker_id}]\n{item.content}"
    elif item.speaker_type == "assistant":
        role = "assistant"
        content = item.content
    else:
        raise ProtocolError("This Memory adapter supports only source human and assistant messages")
    output = {"role": role, "content": content}
    if item.time.parse_status == "exact":
        output["timestamp"] = item.time.parsed_iso8601
    return output


class MemoryHttpClient:
    """Minimal /add and /search client with stable idempotency and bounded retries."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 5.0,
        retry_policy: RetryPolicy | None = None,
        opener: Callable = urllib.request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        require(parsed.scheme in ("http", "https") and bool(parsed.hostname), "base_url must be an HTTP(S) origin")
        require(
            parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment,
            "Credentials and query values are forbidden in base_url",
        )
        require(type(timeout_seconds) in (float, int) and timeout_seconds > 0, "timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.retry_policy = retry_policy or RetryPolicy()
        self.opener = opener
        self.sleeper = sleeper

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[dict, int, float, str]:
        body = canonical(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        started_at = _now_iso()
        tick = time.perf_counter()
        last_code = "NETWORK_ERROR"
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            retry = False
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    if response.status < 200 or response.status >= 300:
                        last_code = f"HTTP_{response.status}"
                        retry = response.status in self.retry_policy.retry_statuses
                    else:
                        raw = response.read(8 * 1024 * 1024 + 1)
                        if len(raw) > 8 * 1024 * 1024:
                            raise ProtocolError("Response exceeds the 8 MiB protocol limit")
                        parsed = _parse_json(raw)
                        if type(parsed) is not dict:
                            raise ProtocolError("Server response must be a JSON object")
                        duration = (time.perf_counter() - tick) * 1000
                        return parsed, attempt, duration, started_at
            except urllib.error.HTTPError as error:
                status = int(error.code)
                error.close()
                last_code = f"HTTP_{status}"
                retry = status in self.retry_policy.retry_statuses
            except urllib.error.URLError as error:
                reason = getattr(error, "reason", None)
                last_code = "TIMEOUT" if isinstance(reason, TimeoutError) else "NETWORK_ERROR"
                retry = True
            except TimeoutError:
                last_code, retry = "TIMEOUT", True
            except ProtocolError as error:
                raise MemoryHttpError("INVALID_RESPONSE", attempt) from error
            except OSError:
                last_code, retry = "NETWORK_ERROR", True
            if not retry or attempt == self.retry_policy.max_attempts:
                raise MemoryHttpError(last_code, attempt) from None
            if self.retry_policy.backoff_seconds:
                self.sleeper(self.retry_policy.backoff_seconds * attempt)
        raise MemoryHttpError(last_code, self.retry_policy.max_attempts)

    def add(self, payload: dict) -> tuple[bool, str | None, int, float, str]:
        required = {"request_id", "user_id", "session_id", "messages"}
        require(type(payload) is dict and set(payload) == required, "Add payload fields do not match protocol")
        for name in ("request_id", "user_id", "session_id"):
            text(payload[name], name)
        require(type(payload["messages"]) is list and bool(payload["messages"]), "Add messages must be nonempty")
        for item in payload["messages"]:
            require(type(item) is dict and set(item) <= {"role", "content", "timestamp"}, "Invalid Add message fields")
            require({"role", "content"} <= set(item), "Add message needs role and content")
            require(item["role"] in ("user", "assistant") and type(item["content"]) is str, "Invalid Add message")
            if "timestamp" in item:
                require(type(item["timestamp"]) is str, "Invalid Add timestamp")
        result, attempts, duration, started = self._request("POST", "/add", payload)
        try:
            require(set(result) <= {"acknowledged", "backend_state_ref"}, "Invalid Add response fields")
            require(result.get("acknowledged") is True, "Add was not explicitly acknowledged")
            state_ref = result.get("backend_state_ref")
            require(state_ref is None or type(state_ref) is str, "Invalid backend_state_ref")
        except SchemaError as error:
            raise MemoryHttpError("INVALID_RESPONSE", attempts) from error
        return True, state_ref, attempts, duration, started

    def search(
        self,
        *,
        run_id: str,
        question_id: str,
        method_id: str,
        namespace: str,
        query: str,
        top_k: int,
        options: dict | None = None,
    ) -> RetrievalArtifact:
        request_obj = RetrievalRequest(namespace, query, top_k, options or {})
        payload = request_obj.to_dict()
        result, attempts, duration, started = self._request("POST", "/search", payload)
        try:
            require(set(result) == {"data"} and type(result["data"]) is list, "Invalid Search response")
            require(len(result["data"]) <= top_k, "Search returned more than top_k hits")
            hits = []
            seen = set()
            for rank, item in enumerate(result["data"], start=1):
                require(type(item) is dict and set(item) <= {"id", "content", "score"}, "Invalid Search hit fields")
                require("id" in item and "content" in item, "Search hit needs id and content")
                text(item["id"], "hit id")
                text(item["content"], "hit content")
                require(item["id"] not in seen, "Duplicate hit ID")
                seen.add(item["id"])
                score = item.get("score")
                require(score is None or type(score) in (int, float), "Invalid hit score")
                from .artifacts import RetrievalHit

                hits.append(RetrievalHit(item["id"], rank, item["content"], score, (), {}))
        except (SchemaError, KeyError, TypeError) as error:
            raise MemoryHttpError("INVALID_RESPONSE", attempts) from error
        return RetrievalArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            retrieval_id="ret-"
            + digest({"run_id": run_id, "question_id": question_id, "method_id": method_id, "request": payload})[:32],
            run_id=run_id,
            question_id=question_id,
            method_id=method_id,
            namespace=namespace,
            request=request_obj,
            hits=tuple(hits),
            status="ok",
            attempts=attempts,
            usage=Usage(None, None, None, None),
            timing=Timing(started, duration),
        )

    def verify_namespace(self, namespace: str) -> str:
        """Use the optional adapter health extension before resuming partial runs."""
        text(namespace, "namespace")
        query = urllib.parse.urlencode({"user_id": namespace})
        result, _, _, _ = self._request("GET", "/state?" + query)
        try:
            require(
                set(result) == {"backend_state_ref"} and type(result["backend_state_ref"]) is str,
                "Invalid state verification response",
            )
        except SchemaError as error:
            raise MemoryHttpError("INVALID_RESPONSE", 1) from error
        return result["backend_state_ref"]


def add_public_history(
    client: MemoryHttpClient,
    history: HistoryArtifact,
    question: PublicQuestionArtifact,
    *,
    run_id: str,
    method_id: str,
    namespace: str,
) -> tuple[AddReceipt, ...]:
    public_messages = validated_public_prefix(history, question)
    by_session: dict[str, list[SourceMessageArtifact]] = {}
    for item in public_messages:
        by_session.setdefault(item.session_id, []).append(item)
    receipts = []
    for session_id, messages in by_session.items():
        payload_seed = {
            "namespace": namespace,
            "session_id": session_id,
            "messages": [_transport_message(item) for item in messages],
        }
        batch_id = "batch-" + digest(payload_seed)[:24]
        request_id = "req-" + digest({"run_id": run_id, "method_id": method_id, "batch_id": batch_id})[:32]
        payload = {
            "request_id": request_id,
            "user_id": namespace,
            "session_id": session_id,
            "messages": payload_seed["messages"],
        }
        acknowledged, state_ref, attempts, duration, started = client.add(payload)
        receipts.append(
            AddReceipt(
                schema_version=ARTIFACT_SCHEMA_VERSION,
                run_id=run_id,
                method_id=method_id,
                namespace=namespace,
                history_prefix_sha256=question.allowed_history.prefix_sha256,
                batch_id=batch_id,
                payload_sha256=digest(payload),
                acknowledged=acknowledged,
                backend_state_ref=state_ref,
                attempts=attempts,
                usage=Usage(None, None, None, None),
                timing=Timing(started, duration),
            )
        )
    return tuple(receipts)
