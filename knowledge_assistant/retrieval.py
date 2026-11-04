"""知识库检索核心：jieba 分词 + BM25 倒排索引。

供在线问答（qa_assistant）和向量混合检索（hybrid_retriever）共用，
不再提供独立的 HTTP 服务。
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Any

from config import load_env

ROOT = Path(__file__).resolve().parents[1]
load_env()
INDEX_PATH = Path(os.getenv("KB_INDEX_PATH", str(ROOT / "knowledge_base_output" / "local_index.json")))


def load_chunks() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


# ---- 分词：jieba（搜索引擎模式）+ 停用词 ----
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jieba

jieba.setLogLevel(logging.WARNING)

STOPWORDS = frozenset(
    "的 了 是 在 和 与 及 或 也 就 都 而 其 这 那 有 为 以 于 对 从 到 被 把 吗 呢 吧 啊 之 者 所 如 若 则 且 等 "
    "我 你 他 她 它 我们 你们 他们 什么 怎么 如何 哪些 多少 为什么 一个 一种 这个 那个 这些 那些 "
    "以及 并且 因为 所以 但是 如果 虽然 然后 还是 就是 不是 没有 已经 可以".split()
)


def terms(text: str) -> list[str]:
    """jieba 搜索引擎模式分词，去掉停用词和纯标点。"""
    text = (text or "").lower()
    tokens = jieba.lcut_for_search(text)
    result: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token in STOPWORDS:
            continue
        if not re.search(r"[0-9a-z\u4e00-\u9fff]", token):
            continue
        result.append(token)
    return result


# ---- BM25 倒排索引 ----
K1 = 1.2
B = 0.75


class _Bm25Index:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.postings: dict[str, dict[int, int]] = {}
        self.lengths: list[int] = []
        for index, item in enumerate(chunks):
            tokens = terms(item.get("content", ""))
            self.lengths.append(len(tokens))
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                self.postings.setdefault(token, {})[index] = tf
        self.total = len(chunks) or 1
        self.avgdl = (sum(self.lengths) / self.total) or 1.0
        self.df = {token: len(postings) for token, postings in self.postings.items()}

    @classmethod
    def from_state(cls, chunks: list[dict[str, Any]], state: dict[str, Any]) -> "_Bm25Index":
        index = cls.__new__(cls)
        index.chunks = chunks
        index.postings = state["postings"]
        index.lengths = state["lengths"]
        index.total = state["total"]
        index.avgdl = state["avgdl"]
        index.df = state["df"]
        return index

    def state(self) -> dict[str, Any]:
        return {
            "postings": self.postings,
            "lengths": self.lengths,
            "total": self.total,
            "avgdl": self.avgdl,
            "df": self.df,
        }

    def idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        return math.log(1 + (self.total - df + 0.5) / (df + 0.5))

    def raw_scores(self, query: str) -> list[float]:
        scores = [0.0] * len(self.chunks)
        for token in set(terms(query)):
            postings = self.postings.get(token)
            if not postings:
                continue
            idf = self.idf(token)
            for index, tf in postings.items():
                denominator = tf + K1 * (1 - B + B * self.lengths[index] / self.avgdl)
                scores[index] += idf * tf * (K1 + 1) / denominator
        return scores


_CACHE: dict[str, Any] = {"key": None, "index": None}
CACHE_PATH = INDEX_PATH.parent / "bm25_cache.pkl"


def bm25_index() -> _Bm25Index:
    key = (INDEX_PATH.stat().st_mtime_ns, INDEX_PATH.stat().st_size) if INDEX_PATH.exists() else None
    if _CACHE["key"] == key and _CACHE["index"] is not None:
        return _CACHE["index"]
    chunks = load_chunks()
    index = None
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("key") == key:
                index = _Bm25Index.from_state(chunks, cached["state"])
        except Exception:
            index = None
    if index is None:
        index = _Bm25Index(chunks)
        try:
            with CACHE_PATH.open("wb") as handle:
                pickle.dump({"key": key, "state": index.state()}, handle)
        except Exception:
            pass
    _CACHE["index"] = index
    _CACHE["key"] = key
    return index


def saturate(raw: float) -> float:
    """把无上限的 BM25 分数压到 0~1，便于融合和阈值判断。"""
    return raw / (raw + 4.0) if raw > 0 else 0.0


def score(query: str, content: str) -> float:
    """单篇内容的 BM25 分（用全局 idf 与平均长度），供混合检索的 0.3 权重使用。"""
    index = bm25_index()
    tokens = terms(content)
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    raw = 0.0
    for token in set(terms(query)):
        tf = counts.get(token)
        if not tf:
            continue
        denominator = tf + K1 * (1 - B + B * len(tokens) / index.avgdl)
        raw += index.idf(token) * tf * (K1 + 1) / denominator
    return saturate(raw)


def bm25_search(query: str, limit: int = 20) -> list[tuple[int, dict, float]]:
    """独立的 BM25 全量召回：返回 [(chunk 下标, chunk, BM25 原始分)]。

    RRF 需要两路各自出排名，所以词法必须能自己召回，而不是只给向量结果补分。
    518 个块全量打合约 0.2ms，成本可以忽略。
    """
    index = bm25_index()
    scores = index.raw_scores(query)
    ranked = sorted(range(len(scores)), key=lambda position: scores[position], reverse=True)
    results: list[tuple[int, dict, float]] = []
    for position in ranked[:limit]:
        if scores[position] <= 0:
            break
        results.append((position, index.chunks[position], scores[position]))
    return results
