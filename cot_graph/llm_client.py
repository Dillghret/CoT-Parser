from __future__ import annotations

import os
from typing import Optional

import certifi
import httpx

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def resolve_api_key(explicit: Optional[str] = None) -> str:
    key = (explicit or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing API key. Set DASHSCOPE_API_KEY (recommended) or pass --api_key.")
    return key


def make_openai_client(*, base_url: str = DEFAULT_BASE_URL, api_key: Optional[str] = None, timeout_sec: float = 180.0):
    if OpenAI is None:
        raise RuntimeError("Please `pip install openai==1.*` first.")

    key = resolve_api_key(api_key)

    transport = httpx.HTTPTransport(retries=0, http2=False)
    http_client = httpx.Client(
        verify=certifi.where(),
        proxies=None,
        timeout=httpx.Timeout(timeout_sec, connect=30.0),
        transport=transport,
        trust_env=False,
    )
    return OpenAI(base_url=base_url, api_key=key, http_client=http_client)
