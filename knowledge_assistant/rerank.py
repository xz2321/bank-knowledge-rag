"""Qwen(DashScope) 文本重排，失败时回退到原始顺序。

DashScope 的重排接口不在 OpenAI 兼容层里，走原生地址：
POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
"""
from __future__ import annotations

import os

import requests

DEFAULT_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


def enabled() -> bool:
    return bool(os.getenv("RERANK_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY"))


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]] | None:
    """返回 [(原始下标, 相关性分数)]，按相关性从高到低；任何失败都返回 None。"""
    key = os.getenv("RERANK_API_KEY") or os.getenv("QWEN_API_KEY")
    if os.getenv("RERANK_ENABLED", "1") == "0" or not key or not documents:
        return None
    url = os.getenv("RERANK_URL", DEFAULT_URL)
    payload = {
        "model": os.getenv("RERANK_MODEL", "qwen3.7-text-rerank"),
        "input": {"query": query, "documents": documents},
        "parameters": {
            "top_n": max(1, min(top_n or len(documents), len(documents))),
            "return_documents": False,
        },
    }
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=float(os.getenv("RERANK_TIMEOUT", "20")),
        )
        response.raise_for_status()
        results = response.json()["output"]["results"]
        ranked = [(int(item["index"]), float(item.get("relevance_score", 0.0))) for item in results]
        return sorted(ranked, key=lambda pair: pair[1], reverse=True)
    except Exception as exc:  # 重排是增强项，失败不影响主流程
        print(f"重排不可用（{exc}），按原顺序返回")
        return None
