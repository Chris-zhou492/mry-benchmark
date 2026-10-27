"""HTTP adapter for a custom memory backend; the default backend is a demo only."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import re
import secrets
from threading import RLock
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, StrictInt


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1)
    timestamp: StrictInt | None = None


class AddInput(BaseModel):
    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)


class SearchInput(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: StrictInt = Field(gt=0)
    options: list[str] | None = None


class DemoMemory:
    """In-process keyword example. Restarting the process loses all its data."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def add(self, *, request_id: str, user_id: str, session_id: str, messages: list[dict]) -> None:
        # Replace this method, or supply your own class with --backend module:Class.
        # A persistent backend must make repeated request_id writes idempotent itself.
        for index, message in enumerate(messages):
            ident = "mem_" + hashlib.sha256(f"{request_id}:{index}".encode()).hexdigest()[:24]
            self.rows[ident] = dict(id=ident, user_id=user_id, session_id=session_id, content=message["content"])

    def search(self, *, user_id: str, query: str, top_k: int, options: list[str] | None = None) -> list[dict]:
        # Return evidence, not a generated answer. options can be ignored by your method.
        terms = set(re.findall(r"\w+", query.lower()))
        hits = []
        for row in self.rows.values():
            if row["user_id"] != user_id:
                continue
            score = len(terms & set(re.findall(r"\w+", row["content"].lower())))
            if score:
                hits.append({"id": row["id"], "content": row["content"], "score": float(score)})
        return sorted(hits, key=lambda hit: (-hit["score"], hit["id"]))[:top_k]


def create_app(backend: Any, *, api_key: str) -> FastAPI:
    if not api_key:
        raise ValueError("Set CHALLENGE_API_KEY before starting the adapter")
    for method in ("add", "search"):
        implementation = getattr(backend, method, None)
        if not callable(implementation) or inspect.iscoroutinefunction(implementation):
            raise ValueError(f"Backend must implement synchronous {method}(); wrap async SDK calls in the backend")
    app = FastAPI(title="Custom Memory adapter")
    lock = RLock()
    requests: dict[str, str] = {}

    def authenticate(x_api_key: str | None = Header(default=None)) -> None:
        if not x_api_key or not secrets.compare_digest(x_api_key.encode(), api_key.encode()):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.post("/add", dependencies=[Depends(authenticate)])
    def add(payload: AddInput) -> dict:
        values = payload.model_dump(exclude_none=True)
        fingerprint = hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with lock:
            previous = requests.get(payload.request_id)
            if previous is not None and previous != fingerprint:
                raise HTTPException(status_code=409, detail="request_id already used with different content")
            if previous is None:
                backend.add(**values)
                requests[payload.request_id] = fingerprint
        return dict(success=True, request_id=payload.request_id, user_id=payload.user_id, session_id=payload.session_id)

    @app.post("/search", dependencies=[Depends(authenticate)])
    def search(payload: SearchInput) -> dict:
        with lock:
            hits = backend.search(**payload.model_dump())
        if not isinstance(hits, list) or len(hits) > payload.top_k:
            raise HTTPException(status_code=502, detail="Backend must return a list of at most top_k hits")
        if any(
            not isinstance(hit, dict) or not isinstance(hit.get("content"), str) or not hit["content"].strip()
            for hit in hits
        ):
            raise HTTPException(status_code=502, detail="Each hit needs non-empty text content")
        return {"data": hits}

    return app


def main() -> None:
    load_dotenv(interpolate=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="memory_adapter:DemoMemory", help="Importable module:Class with a no-argument constructor"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    module, name = args.backend.split(":", 1)
    backend = getattr(importlib.import_module(module), name)()
    uvicorn.run(create_app(backend, api_key=os.environ.get("CHALLENGE_API_KEY", "")), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
