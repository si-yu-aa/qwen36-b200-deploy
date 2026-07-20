from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import httpx


SCRIPT = Path(__file__).parents[1] / "scripts" / "qwen36-timing-proxy.py"


def load_proxy_module():
    spec = importlib.util.spec_from_file_location("qwen36_timing_proxy_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.api_key = "test-secret"
    return module


def request(module, path: str, headers: dict[str, str] | None = None) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=module.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return await client.get(path, headers=headers)

    return asyncio.run(send())


def test_generic_proxy_route_rejects_missing_or_bad_bearer_token() -> None:
    module = load_proxy_module()

    missing = request(module, "/v1/models")
    bad = request(
        module,
        "/v1/models",
        headers={"Authorization": "Bearer wrong-secret"},
    )

    assert missing.status_code == 401
    assert bad.status_code == 401


def test_timing_route_rejects_missing_bearer_token() -> None:
    module = load_proxy_module()

    response = request(module, "/_timing/health")

    assert response.status_code == 401
