#!/usr/bin/env python3
"""Benchmark real chat payloads over an OpenAI-compatible streaming endpoint."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import httpx


METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+)$"
)
LABEL_RE = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            messages = json.loads(row["prompts"])
            tools_raw = row.get("tools", "").strip()
            tools = json.loads(tools_raw) if tools_raw else None
            rows.append(
                {
                    "source_index": index,
                    "messages": messages,
                    "tools": tools or None,
                    "trace_id": row.get("trace_id", ""),
                }
            )
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def parse_labels(raw: str | None) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    labels = []
    for key, value in LABEL_RE.findall(raw):
        labels.append((key, bytes(value, "utf-8").decode("unicode_escape")))
    return tuple(sorted(labels))


def parse_metrics(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    parsed: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if match:
            name, labels_raw, value = match.groups()
            parsed[(name, parse_labels(labels_raw))] = float(value)
    return parsed


def metric_total(
    snapshot: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
    required_labels: dict[str, str] | None = None,
) -> float:
    total = 0.0
    for (sample_name, label_pairs), value in snapshot.items():
        if sample_name != name:
            continue
        labels = dict(label_pairs)
        if required_labels and any(labels.get(k) != v for k, v in required_labels.items()):
            continue
        total += value
    return total


def delta_metric(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    name: str,
    required_labels: dict[str, str] | None = None,
) -> float:
    return metric_total(after, name, required_labels) - metric_total(
        before, name, required_labels
    )


def histogram_mean(
    before: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    after: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    base: str,
    required_labels: dict[str, str] | None = None,
) -> float | None:
    count = delta_metric(before, after, f"{base}_count", required_labels)
    total = delta_metric(before, after, f"{base}_sum", required_labels)
    return total / count if count > 0 else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[rank]


def summarize_ms(values_s: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values_s) * 1000 if values_s else None,
        "p50": percentile(values_s, 0.50) * 1000 if values_s else None,
        "p95": percentile(values_s, 0.95) * 1000 if values_s else None,
        "max": max(values_s) * 1000 if values_s else None,
    }


async def scrape_metrics(client: httpx.AsyncClient, metrics_url: str) -> dict:
    response = await client.get(metrics_url)
    response.raise_for_status()
    return parse_metrics(response.text)


async def run_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    item: dict[str, Any],
    sequence: int,
    max_tokens: int,
    cache_salt: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": item["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max_tokens,
        "cache_salt": cache_salt,
    }
    if item["tools"]:
        payload["tools"] = item["tools"]
        payload["tool_choice"] = "auto"

    started = time.perf_counter()
    headers_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    response_id: str | None = None
    completion_tokens: int | None = None
    output_chars = 0
    wire_chars = 0

    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            headers_at = time.perf_counter()
            response.raise_for_status()
            async for line in response.aiter_lines():
                wire_chars += len(line)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished_at = time.perf_counter()
                    break
                if not data:
                    continue
                chunk = json.loads(data)
                response_id = response_id or chunk.get("id")
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                meaningful = False
                for field in ("content", "reasoning_content"):
                    value = delta.get(field)
                    if value:
                        meaningful = True
                        output_chars += len(value)
                if delta.get("tool_calls"):
                    meaningful = True
                if meaningful and first_token_at is None:
                    first_token_at = time.perf_counter()
        if finished_at is None:
            finished_at = time.perf_counter()
        if first_token_at is None:
            first_token_at = headers_at or finished_at
        return {
            "sequence": sequence,
            "source_index": item["source_index"],
            "source_trace_id": item["trace_id"],
            "response_id": response_id,
            "status": "ok",
            "headers_s": (headers_at or started) - started,
            "ttft_s": first_token_at - started,
            "stream_s": finished_at - first_token_at,
            "e2e_s": finished_at - started,
            "completion_tokens": completion_tokens,
            "output_chars": output_chars,
            "wire_chars": wire_chars,
        }
    except Exception as exc:
        ended = time.perf_counter()
        return {
            "sequence": sequence,
            "source_index": item["source_index"],
            "source_trace_id": item["trace_id"],
            "response_id": response_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "e2e_s": ended - started,
        }


async def run_level(
    *,
    base_url: str,
    api_key: str,
    model: str,
    workload: list[dict[str, Any]],
    concurrency: int,
    max_tokens: int,
) -> dict[str, Any]:
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=120,
    )
    timeout = httpx.Timeout(connect=30, read=240, write=60, pool=60)
    headers = {"Authorization": f"Bearer {api_key}"}
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    origin = base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[:-3]
    metrics_url = f"{origin}/metrics"
    models_url = f"{base_url.rstrip('/')}/models"

    async with httpx.AsyncClient(
        headers=headers,
        limits=limits,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        # Establish the same number of warm HTTPS connections used by the run.
        warmups = await asyncio.gather(
            *(client.get(models_url) for _ in range(concurrency))
        )
        for response in warmups:
            response.raise_for_status()

        before = await scrape_metrics(client, metrics_url)
        semaphore = asyncio.Semaphore(concurrency)
        cache_salt = f"business-bench-c{concurrency}-{time.time_ns()}"

        async def guarded(sequence: int, item: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await run_request(
                    client,
                    endpoint,
                    model,
                    item,
                    sequence,
                    max_tokens,
                    cache_salt,
                )

        wall_started = time.perf_counter()
        results = await asyncio.gather(
            *(guarded(index, item) for index, item in enumerate(workload))
        )
        wall_s = time.perf_counter() - wall_started
        await asyncio.sleep(0.5)
        after = await scrape_metrics(client, metrics_url)

    ok = [item for item in results if item["status"] == "ok"]
    errors = [item for item in results if item["status"] != "ok"]
    client_ttft = [item["ttft_s"] for item in ok]
    client_stream = [item["stream_s"] for item in ok]
    client_e2e = [item["e2e_s"] for item in ok]
    client_headers = [item["headers_s"] for item in ok]

    server_ttft = histogram_mean(
        before, after, "sglang:time_to_first_token_seconds"
    )
    server_e2e = histogram_mean(
        before, after, "sglang:e2e_request_latency_seconds"
    )
    server_queue = histogram_mean(before, after, "sglang:queue_time_seconds")
    server_prefill = histogram_mean(
        before,
        after,
        "sglang:per_stage_req_latency_seconds",
        {"stage": "prefill_forward"},
    )
    request_count = delta_metric(
        before, after, "sglang:e2e_request_latency_seconds_count"
    )
    prompt_tokens = delta_metric(before, after, "sglang:prompt_tokens_total")
    generation_tokens = delta_metric(
        before, after, "sglang:generation_tokens_total"
    )
    mean_client_ttft = statistics.fmean(client_ttft) if client_ttft else None
    mean_client_e2e = statistics.fmean(client_e2e) if client_e2e else None
    server_decode = (
        server_e2e - server_ttft
        if server_e2e is not None and server_ttft is not None
        else None
    )

    return {
        "concurrency": concurrency,
        "cache_salt": cache_salt,
        "requests": len(workload),
        "succeeded": len(ok),
        "failed": len(errors),
        "wall_s": wall_s,
        "request_throughput_per_s": len(ok) / wall_s if wall_s else None,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "output_tokens_per_s": generation_tokens / wall_s if wall_s else None,
        "server_observed_requests": request_count,
        "client_headers_ms": summarize_ms(client_headers),
        "client_ttft_ms": summarize_ms(client_ttft),
        "client_stream_ms": summarize_ms(client_stream),
        "client_e2e_ms": summarize_ms(client_e2e),
        "server_mean_ms": {
            "queue": server_queue * 1000 if server_queue is not None else None,
            "prefill_forward": server_prefill * 1000 if server_prefill is not None else None,
            "ttft": server_ttft * 1000 if server_ttft is not None else None,
            "decode_after_first_token": server_decode * 1000
            if server_decode is not None
            else None,
            "e2e": server_e2e * 1000 if server_e2e is not None else None,
        },
        "residual_mean_ms": {
            "before_first_token": (mean_client_ttft - server_ttft) * 1000
            if mean_client_ttft is not None and server_ttft is not None
            else None,
            "total": (mean_client_e2e - server_e2e) * 1000
            if mean_client_e2e is not None and server_e2e is not None
            else None,
            "stream_vs_decode": (statistics.fmean(client_stream) - server_decode)
            * 1000
            if client_stream and server_decode is not None
            else None,
        },
        "errors": errors,
        "per_request": results,
    }


async def main() -> None:
    args = parse_args()
    source_rows = load_rows(Path(args.csv))
    workload = [source_rows[index % len(source_rows)] for index in range(args.requests)]
    api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("API key file is empty")
    concurrency_levels = [int(value) for value in args.concurrency.split(",")]

    output: dict[str, Any] = {
        "source_csv": str(Path(args.csv).resolve()),
        "source_rows": len(source_rows),
        "requests_per_level": len(workload),
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "stream": True,
        "concurrency_levels": concurrency_levels,
        "levels": [],
    }
    for concurrency in concurrency_levels:
        level = await run_level(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            workload=workload,
            concurrency=concurrency,
            max_tokens=args.max_tokens,
        )
        output["levels"].append(level)
        print("SUMMARY " + json.dumps({k: v for k, v in level.items() if k not in {"per_request", "errors"}}, ensure_ascii=False), flush=True)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE {destination}")


if __name__ == "__main__":
    asyncio.run(main())
