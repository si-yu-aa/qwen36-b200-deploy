#!/usr/bin/env python3
"""Measure business streaming latency through an SSH timing proxy."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from bench_business_stream import (
    delta_metric,
    histogram_mean,
    load_rows,
    scrape_metrics,
    summarize_ms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--model", default="Qwen3.6-35B-A3B")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--clock-samples", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def ns_delta_ms(end_ns: int, start_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000


def summarize_numbers(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(q: float) -> float:
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    return {
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(values),
    }


async def clock_sample(client: httpx.AsyncClient, clock_url: str) -> dict[str, float]:
    client_send_ns = time.time_ns()
    response = await client.get(clock_url)
    client_receive_ns = time.time_ns()
    response.raise_for_status()
    server_receive_ns = int(response.json()["server_received_ns"])
    server_send_ns = int(response.headers["x-server-sent-ns"])
    offset_ns = (
        (server_receive_ns - client_send_ns)
        + (server_send_ns - client_receive_ns)
    ) / 2
    network_delay_ns = (client_receive_ns - client_send_ns) - (
        server_send_ns - server_receive_ns
    )
    return {
        "offset_server_minus_client_ns": offset_ns,
        "round_trip_delay_ns": network_delay_ns,
    }


def estimate_clock(samples: list[dict[str, float]]) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda value: value["round_trip_delay_ns"])
    selected = ordered[: min(5, len(ordered))]
    offsets = [value["offset_server_minus_client_ns"] for value in selected]
    offset_ns = statistics.median(offsets)
    best_rtt_ns = ordered[0]["round_trip_delay_ns"]
    return {
        "offset_server_minus_client_ns": offset_ns,
        "best_round_trip_ms": best_rtt_ns / 1_000_000,
        "one_way_uncertainty_bound_ms": best_rtt_ns / 2_000_000,
        "selected_samples": len(selected),
        "all_round_trip_ms": [
            value["round_trip_delay_ns"] / 1_000_000 for value in ordered
        ],
    }


async def run_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    item: dict[str, Any],
    sequence: int,
    max_tokens: int,
    cache_salt: str,
    run_id: str,
) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
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
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    request_headers = {
        "content-type": "application/json",
        "x-request-id": request_id,
        "x-benchmark-run-id": run_id,
        "x-benchmark-sequence": str(sequence),
    }

    started_perf_ns = time.perf_counter_ns()
    started_wall_ns = time.time_ns()
    headers_perf_ns: int | None = None
    headers_wall_ns: int | None = None
    first_token_perf_ns: int | None = None
    first_token_wall_ns: int | None = None
    finished_perf_ns: int | None = None
    finished_wall_ns: int | None = None
    response_id: str | None = None
    completion_tokens: int | None = None
    output_chars = 0
    wire_bytes = 0

    try:
        async with client.stream(
            "POST", endpoint, content=body, headers=request_headers
        ) as response:
            headers_perf_ns = time.perf_counter_ns()
            headers_wall_ns = time.time_ns()
            response.raise_for_status()
            async for line in response.aiter_lines():
                wire_bytes += len(line.encode("utf-8")) + 1
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished_perf_ns = time.perf_counter_ns()
                    finished_wall_ns = time.time_ns()
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
                if meaningful and first_token_perf_ns is None:
                    first_token_perf_ns = time.perf_counter_ns()
                    first_token_wall_ns = time.time_ns()
        if finished_perf_ns is None:
            finished_perf_ns = time.perf_counter_ns()
            finished_wall_ns = time.time_ns()
        if first_token_perf_ns is None:
            first_token_perf_ns = headers_perf_ns or finished_perf_ns
            first_token_wall_ns = headers_wall_ns or finished_wall_ns
        return {
            "request_id": request_id,
            "sequence": sequence,
            "source_index": item["source_index"],
            "source_trace_id": item["trace_id"],
            "response_id": response_id,
            "status": "ok",
            "request_body_bytes": len(body),
            "client_started_wall_ns": started_wall_ns,
            "client_headers_wall_ns": headers_wall_ns,
            "client_first_token_wall_ns": first_token_wall_ns,
            "client_finished_wall_ns": finished_wall_ns,
            "headers_ms": ns_delta_ms(headers_perf_ns or started_perf_ns, started_perf_ns),
            "ttft_ms": ns_delta_ms(first_token_perf_ns, started_perf_ns),
            "stream_ms": ns_delta_ms(finished_perf_ns, first_token_perf_ns),
            "e2e_ms": ns_delta_ms(finished_perf_ns, started_perf_ns),
            "completion_tokens": completion_tokens,
            "output_chars": output_chars,
            "wire_bytes": wire_bytes,
        }
    except Exception as exc:
        ended_perf_ns = time.perf_counter_ns()
        return {
            "request_id": request_id,
            "sequence": sequence,
            "source_index": item["source_index"],
            "source_trace_id": item["trace_id"],
            "response_id": response_id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "e2e_ms": ns_delta_ms(ended_perf_ns, started_perf_ns),
        }


def add_timing_segments(
    client_record: dict[str, Any],
    proxy_record: dict[str, Any],
    offset_ns: float,
) -> None:
    client_record["proxy_timing"] = proxy_record

    def server_as_client_ns(field: str) -> float:
        return float(proxy_record[field]) - offset_ns

    client_start = int(client_record["client_started_wall_ns"])
    client_first = int(client_record["client_first_token_wall_ns"])
    client_finished = int(client_record["client_finished_wall_ns"])
    headers_received = int(proxy_record["request_headers_received_wall_ns"])
    body_complete = int(proxy_record["request_body_complete_wall_ns"])
    upstream_start = int(proxy_record["upstream_request_start_wall_ns"])
    first_ready = int(proxy_record["first_token_ready_wall_ns"])
    done_ready = int(proxy_record["done_ready_wall_ns"])
    first_sent = int(
        proxy_record.get("first_token_send_complete_wall_ns", first_ready)
    )
    done_sent = int(proxy_record.get("done_send_complete_wall_ns", done_ready))

    remote_service_ms = ns_delta_ms(done_sent, body_complete)
    client_record["segments_ms"] = {
        "uplink_to_b200_headers_est": (
            server_as_client_ns("request_headers_received_wall_ns") - client_start
        )
        / 1_000_000,
        "uplink_to_b200_full_body_est": (
            server_as_client_ns("request_body_complete_wall_ns") - client_start
        )
        / 1_000_000,
        "b200_request_body_receive": ns_delta_ms(body_complete, headers_received),
        "b200_proxy_dispatch": ns_delta_ms(upstream_start, body_complete),
        "b200_body_complete_to_first_token": ns_delta_ms(first_ready, body_complete),
        "b200_first_token_proxy_send": ns_delta_ms(first_sent, first_ready),
        "first_token_downlink_est": (
            client_first - (first_sent - offset_ns)
        )
        / 1_000_000,
        "b200_generation_first_to_done": ns_delta_ms(done_ready, first_ready),
        "b200_done_proxy_send": ns_delta_ms(done_sent, done_ready),
        "last_token_downlink_est": (
            client_finished - (done_sent - offset_ns)
        )
        / 1_000_000,
        "b200_service_body_to_done": remote_service_ms,
        "round_trip_link_and_client_overhead": client_record["e2e_ms"]
        - remote_service_ms,
        "streaming_network_effect": client_record["stream_ms"]
        - ns_delta_ms(done_ready, first_ready),
    }


async def fetch_run_timings(
    client: httpx.AsyncClient, timing_root: str, run_id: str, expected: int
) -> list[dict[str, Any]]:
    url = f"{timing_root}/_timing/run/{run_id}"
    for _ in range(10):
        response = await client.get(url)
        response.raise_for_status()
        records = response.json()["records"]
        if len(records) >= expected:
            return records
        await asyncio.sleep(0.1)
    return records


async def run_level(
    *,
    base_url: str,
    api_key: str,
    model: str,
    workload: list[dict[str, Any]],
    concurrency: int,
    max_tokens: int,
    clock_samples: int,
) -> dict[str, Any]:
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=180,
    )
    timeout = httpx.Timeout(connect=30, read=240, write=60, pool=60)
    auth_headers = {"Authorization": f"Bearer {api_key}"}
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    timing_root = base_url.rstrip("/")
    if timing_root.endswith("/v1"):
        timing_root = timing_root[:-3]
    metrics_url = f"{timing_root}/metrics"
    models_url = f"{base_url.rstrip('/')}/models"
    clock_url = f"{timing_root}/_timing/clock"
    run_id = f"business-ssh-c{concurrency}-{uuid.uuid4().hex}"

    async with httpx.AsyncClient(
        headers=auth_headers,
        limits=limits,
        timeout=timeout,
    ) as client:
        warmups = await asyncio.gather(
            *(client.get(models_url) for _ in range(concurrency))
        )
        for response in warmups:
            response.raise_for_status()

        clock_before = [
            await clock_sample(client, clock_url) for _ in range(clock_samples)
        ]
        before = await scrape_metrics(client, metrics_url)
        cache_salt = f"business-timed-ssh-c{concurrency}-{time.time_ns()}"
        semaphore = asyncio.Semaphore(concurrency)

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
                    run_id,
                )

        wall_started = time.perf_counter()
        results = await asyncio.gather(
            *(guarded(index, item) for index, item in enumerate(workload))
        )
        wall_s = time.perf_counter() - wall_started
        await asyncio.sleep(0.5)
        after = await scrape_metrics(client, metrics_url)
        clock_after = [
            await clock_sample(client, clock_url) for _ in range(clock_samples)
        ]
        proxy_records = await fetch_run_timings(
            client, timing_root, run_id, len(workload)
        )

    clock = estimate_clock(clock_before + clock_after)
    proxy_by_id = {value["request_id"]: value for value in proxy_records}
    for result in results:
        if result["status"] != "ok":
            continue
        proxy_record = proxy_by_id.get(result["request_id"])
        if proxy_record:
            add_timing_segments(
                result,
                proxy_record,
                float(clock["offset_server_minus_client_ns"]),
            )

    ok = [value for value in results if value["status"] == "ok"]
    timed = [value for value in ok if "segments_ms" in value]
    errors = [value for value in results if value["status"] != "ok"]

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
    prompt_tokens = delta_metric(before, after, "sglang:prompt_tokens_total")
    generation_tokens = delta_metric(
        before, after, "sglang:generation_tokens_total"
    )

    segment_names = list(timed[0]["segments_ms"]) if timed else []
    segment_summary = {
        name: summarize_numbers(
            [float(value["segments_ms"][name]) for value in timed]
        )
        for name in segment_names
    }
    body_to_first = segment_summary.get("b200_body_complete_to_first_token", {}).get(
        "mean"
    )
    sglang_ttft_ms = server_ttft * 1000 if server_ttft is not None else None

    return {
        "concurrency": concurrency,
        "run_id": run_id,
        "cache_salt": cache_salt,
        "requests": len(workload),
        "succeeded": len(ok),
        "timed": len(timed),
        "failed": len(errors),
        "wall_s": wall_s,
        "request_throughput_per_s": len(ok) / wall_s if wall_s else None,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "output_tokens_per_s": generation_tokens / wall_s if wall_s else None,
        "request_body_bytes": summarize_numbers(
            [float(value["request_body_bytes"]) for value in ok]
        ),
        "client_ms": {
            "headers": summarize_numbers([float(value["headers_ms"]) for value in ok]),
            "ttft": summarize_numbers([float(value["ttft_ms"]) for value in ok]),
            "stream": summarize_numbers([float(value["stream_ms"]) for value in ok]),
            "e2e": summarize_numbers([float(value["e2e_ms"]) for value in ok]),
        },
        "server_sglang_mean_ms": {
            "queue": server_queue * 1000 if server_queue is not None else None,
            "prefill_forward": server_prefill * 1000
            if server_prefill is not None
            else None,
            "ttft": sglang_ttft_ms,
            "decode_after_first_token": (server_e2e - server_ttft) * 1000
            if server_e2e is not None and server_ttft is not None
            else None,
            "e2e": server_e2e * 1000 if server_e2e is not None else None,
        },
        "b200_pre_sglang_metric_overhead_mean_ms": body_to_first - sglang_ttft_ms
        if body_to_first is not None and sglang_ttft_ms is not None
        else None,
        "segments_ms": segment_summary,
        "clock_calibration": clock,
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
        "transport": "SSH local port forwarding",
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
            clock_samples=args.clock_samples,
        )
        output["levels"].append(level)
        summary = {
            "concurrency": concurrency,
            "succeeded": level["succeeded"],
            "timed": level["timed"],
            "output_tokens_per_s": level["output_tokens_per_s"],
            "client_ms": level["client_ms"],
            "server_sglang_mean_ms": level["server_sglang_mean_ms"],
            "segments_ms": level["segments_ms"],
            "clock_calibration": level["clock_calibration"],
        }
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WROTE {destination}")


if __name__ == "__main__":
    asyncio.run(main())
