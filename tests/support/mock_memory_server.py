"""A local-only HTTP mock implementing the benchmark's explicit adapter contract."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from memory_benchmark.schemas import canonical, digest


class MockMemoryState:
    def __init__(self):
        self.lock = threading.Lock()
        self.messages: dict[str, list[dict]] = {}
        self.idempotency: dict[str, str] = {}
        self.captured: list[dict] = []
        self.failures: Counter[tuple[str, int]] = Counter()
        self.request_counts: Counter[tuple[str, str]] = Counter()

    def fail_next(self, path: str, status: int, count: int = 1):
        with self.lock:
            self.failures[(path, status)] += count

    def take_failure(self, path: str):
        with self.lock:
            for (failed_path, status), count in list(self.failures.items()):
                if failed_path == path and count:
                    self.failures[(failed_path, status)] -= 1
                    return status
        return None

    def snapshot_ref(self, user_id: str) -> str:
        return "state-" + digest(self.messages.get(user_id, []))


class MockMemoryServer:
    def __init__(self):
        self.state = MockMemoryState()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                return

            def _reply(self, status: int, body: dict):
                raw = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_json(self):
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError("bad body size")
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8"))
                if type(body) is not dict:
                    raise ValueError("bad body")
                return raw, body

            def do_GET(self):
                parsed = urlsplit(self.path)
                owner.state.request_counts[("GET", parsed.path)] += 1
                if parsed.path != "/state":
                    self._reply(404, {"error": "not found"})
                    return
                user_id = parse_qs(parsed.query).get("user_id", [""])[0]
                if not user_id:
                    self._reply(400, {"error": "user_id required"})
                    return
                with owner.state.lock:
                    state_ref = owner.state.snapshot_ref(user_id)
                self._reply(200, {"backend_state_ref": state_ref})

            def do_POST(self):
                parsed = urlsplit(self.path)
                try:
                    raw, body = self._read_json()
                except Exception:
                    self._reply(400, {"error": "invalid request"})
                    return
                with owner.state.lock:
                    owner.state.request_counts[("POST", parsed.path)] += 1
                    owner.state.captured.append(
                        {"method": "POST", "path": parsed.path, "body": body, "raw": raw.decode("utf-8")}
                    )
                failure = owner.state.take_failure(parsed.path)
                if failure is not None:
                    self._reply(failure, {"error": "injected mock failure"})
                    return
                if parsed.path == "/add":
                    self._add(body)
                elif parsed.path == "/search":
                    self._search(body)
                else:
                    self._reply(404, {"error": "not found"})

            def _add(self, body):
                if set(body) != {"request_id", "user_id", "session_id", "messages"}:
                    self._reply(400, {"error": "invalid Add shape"})
                    return
                request_id = body["request_id"]
                payload_digest = hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()
                with owner.state.lock:
                    previous = owner.state.idempotency.get(request_id)
                    if previous is not None and previous != payload_digest:
                        self._reply(409, {"error": "idempotency conflict"})
                        return
                    if previous is None:
                        namespace_messages = owner.state.messages.setdefault(body["user_id"], [])
                        for index, item in enumerate(body["messages"]):
                            memory_id = (
                                "mem-"
                                + digest(
                                    {
                                        "user_id": body["user_id"],
                                        "session_id": body["session_id"],
                                        "index": index,
                                        "message": item,
                                    }
                                )[:24]
                            )
                            namespace_messages.append(
                                {
                                    "id": memory_id,
                                    "content": item["content"],
                                    "role": item["role"],
                                    "session_id": body["session_id"],
                                }
                            )
                        owner.state.idempotency[request_id] = payload_digest
                    state_ref = owner.state.snapshot_ref(body["user_id"])
                self._reply(200, {"acknowledged": True, "backend_state_ref": state_ref})

            def _search(self, body):
                if set(body) - {"user_id", "query", "top_k", "options"} or not {"user_id", "query", "top_k"} <= set(
                    body
                ):
                    self._reply(400, {"error": "invalid Search shape"})
                    return
                query_tokens = {token.casefold() for token in body["query"].split() if token}
                with owner.state.lock:
                    memories = list(owner.state.messages.get(body["user_id"], []))
                scored = []
                for item in memories:
                    content_tokens = {token.strip(".,!?;:'\"()[]{}").casefold() for token in item["content"].split()}
                    score = len(query_tokens & content_tokens)
                    if score:
                        scored.append((score, item))
                scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
                data = [
                    {"id": item["id"], "content": item["content"], "score": float(score)}
                    for score, item in scored[: body["top_k"]]
                ]
                self._reply(200, {"data": data})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def clear_namespace(self, namespace: str):
        with self.state.lock:
            self.state.messages.pop(namespace, None)
            # A lost backend namespace also loses the service's request-id state.
            self.state.idempotency = {
                request_id: fingerprint
                for request_id, fingerprint in self.state.idempotency.items()
                if request_id
                not in {
                    item["body"].get("request_id")
                    for item in self.state.captured
                    if item["body"].get("user_id") == namespace
                }
            }
