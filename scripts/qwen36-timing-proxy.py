#!/usr/bin/env python3
"""Low-overhead streaming timing proxy for the local SGLang endpoint."""

from __future__ import annotations

import hmac
import json
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


UPSTREAM = os.environ.get("TIMING_PROXY_UPSTREAM", "http://127.0.0.1:30000")
HOST = os.environ.get("TIMING_PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("TIMING_PROXY_PORT", "30001"))
KEEP_ALIVE_SECONDS = int(os.environ.get("TIMING_PROXY_KEEP_ALIVE", "60"))
API_KEY_FILE = Path(
    os.environ.get("TIMING_PROXY_API_KEY_FILE", "/workspace/.qwen36_api_key")
)
LOG_PATH = Path(
    os.environ.get("TIMING_PROXY_LOG", "/workspace/logs/qwen36-timing.jsonl")
)
MAX_RECORDS = int(os.environ.get("TIMING_PROXY_MAX_RECORDS", "10000"))

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

api_key = ""
upstream_client: httpx.AsyncClient
records: OrderedDict[str, dict[str, Any]] = OrderedDict()


def stamp(record: dict[str, Any], name: str) -> None:
    record[f"{name}_wall_ns"] = time.time_ns()
    record[f"{name}_mono_ns"] = time.perf_counter_ns()


def authorized(request: Request) -> bool:
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {api_key}"
    return bool(api_key) and hmac.compare_digest(supplied, expected)


def require_auth(request: Request) -> None:
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def save_record(record: dict[str, Any]) -> None:
    clean = public_record(record)
    request_id = str(clean["request_id"])
    records[request_id] = clean
    records.move_to_end(request_id)
    while len(records) > MAX_RECORDS:
        records.popitem(last=False)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def scan_sse(record: dict[str, Any], chunk: bytes) -> None:
    pending = record.get("_sse_pending", b"") + chunk
    lines = pending.split(b"\n")
    record["_sse_pending"] = lines.pop()
    for raw_line in lines:
        line = raw_line.rstrip(b"\r")
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            if "done_ready_wall_ns" not in record:
                stamp(record, "done_ready")
            continue
        if not data or "first_token_ready_wall_ns" in record:
            continue
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        choices = payload.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        meaningful = bool(
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("tool_calls")
        )
        if meaningful:
            stamp(record, "first_token_ready")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global api_key, upstream_client
    api_key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(f"Empty API key file: {API_KEY_FILE}")
    limits = httpx.Limits(
        max_connections=128,
        max_keepalive_connections=128,
        keepalive_expiry=300,
    )
    upstream_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=600, write=60, pool=60),
        limits=limits,
    )
    try:
        yield
    finally:
        await upstream_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/_timing/health")
async def health(request: Request) -> dict[str, str]:
    require_auth(request)
    return {"status": "ok", "upstream": UPSTREAM}


@app.get("/_timing/clock")
async def clock(request: Request) -> JSONResponse:
    require_auth(request)
    received_ns = time.time_ns()
    response = JSONResponse({"server_received_ns": received_ns})
    response.headers["x-server-sent-ns"] = str(time.time_ns())
    response.headers["cache-control"] = "no-store"
    return response


@app.get("/_timing/run/{run_id}")
async def timings_for_run(request: Request, run_id: str) -> JSONResponse:
    require_auth(request)
    matched = [
        value for value in records.values() if value.get("run_id") == run_id
    ]
    matched.sort(key=lambda value: int(value.get("sequence", -1)))
    return JSONResponse({"run_id": run_id, "records": matched})


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def proxy(request: Request, path: str) -> Response:
    require_auth(request)
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    run_id = request.headers.get("x-benchmark-run-id")
    sequence_raw = request.headers.get("x-benchmark-sequence")
    record: dict[str, Any] = {
        "request_id": request_id,
        "run_id": run_id,
        "sequence": int(sequence_raw) if sequence_raw is not None else -1,
        "method": request.method,
        "path": f"/{path}",
    }
    stamp(record, "request_headers_received")
    body = await request.body()
    record["request_body_bytes"] = len(body)
    stamp(record, "request_body_complete")

    outgoing_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP | {"host", "content-length"}
    }
    outgoing_headers["x-request-id"] = request_id
    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    upstream_request = upstream_client.build_request(
        request.method,
        url,
        headers=outgoing_headers,
        content=body,
    )
    stamp(record, "upstream_request_start")
    try:
        upstream_response = await upstream_client.send(upstream_request, stream=True)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        stamp(record, "proxy_failed")
        save_record(record)
        raise HTTPException(status_code=502, detail="Upstream request failed") from exc
    stamp(record, "upstream_headers_received")
    record["upstream_status"] = upstream_response.status_code

    response_headers: dict[str, str] = {}
    for key, value in upstream_response.headers.multi_items():
        lowered = key.lower()
        if lowered not in HOP_BY_HOP | {"content-length"}:
            response_headers[key] = value
    response_headers["x-timing-request-id"] = request_id
    response_headers["x-accel-buffering"] = "no"

    content_type = upstream_response.headers.get("content-type", "").lower()
    if not content_type.startswith("text/event-stream"):
        try:
            content = await upstream_response.aread()
            stamp(record, "first_upstream_body")
            record["response_body_bytes"] = len(content)
            stamp(record, "done_ready")
            stamp(record, "done_send_complete")
            stamp(record, "upstream_stream_complete")
        finally:
            await upstream_response.aclose()
            stamp(record, "proxy_stream_finalized")
            save_record(record)
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    async def body_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_raw():
                if not chunk:
                    continue
                if "first_upstream_body_wall_ns" not in record:
                    stamp(record, "first_upstream_body")
                had_first_token = "first_token_ready_wall_ns" in record
                had_done = "done_ready_wall_ns" in record
                scan_sse(record, chunk)
                first_token_in_chunk = (
                    not had_first_token and "first_token_ready_wall_ns" in record
                )
                done_in_chunk = not had_done and "done_ready_wall_ns" in record
                record["response_body_bytes"] = int(
                    record.get("response_body_bytes", 0)
                ) + len(chunk)
                yield chunk
                # Resuming the generator means Starlette/Uvicorn completed its
                # ASGI send call and handed this chunk to the socket transport.
                if first_token_in_chunk:
                    stamp(record, "first_token_send_complete")
                if done_in_chunk:
                    stamp(record, "done_send_complete")
            if "done_ready_wall_ns" not in record:
                stamp(record, "done_ready")
            if "done_send_complete_wall_ns" not in record:
                stamp(record, "done_send_complete")
            stamp(record, "upstream_stream_complete")
        except BaseException as exc:
            record["stream_error"] = f"{type(exc).__name__}: {exc}"
            stamp(record, "stream_failed")
            raise
        finally:
            await upstream_response.aclose()
            stamp(record, "proxy_stream_finalized")
            save_record(record)

    return StreamingResponse(
        body_iterator(),
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        access_log=False,
        log_level="warning",
        timeout_keep_alive=KEEP_ALIVE_SECONDS,
    )
