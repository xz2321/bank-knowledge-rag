"""Optional Qdrant hybrid retriever.

Embedding requires an OpenAI-compatible embeddings endpoint (DeepSeek chat does
not provide embeddings). Set EMBEDDING_BASE_URL, EMBEDDING_API_KEY and
EMBEDDING_MODEL before building the collection.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from retrieval import bm25_search, load_chunks, score
from rerank import rerank

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(ROOT / "knowledge_base_output" / "qdrant")))
COLLECTION = os.getenv("QDRANT_COLLECTION", "bank_knowledge")


def embedding_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
    )


def embed(client, texts: list[str]) -> list[list[float]]:
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return [item.embedding for item in client.embeddings.create(model=model, input=texts).data]


_CLIENT: QdrantClient | None = None


def qdrant_client() -> QdrantClient:
    """复用同一个本地客户端：本地模式同一时间只允许一个进程打开存储目录。"""
    global _CLIENT
    if _CLIENT is None:
        try:
            _CLIENT = QdrantClient(path=str(QDRANT_PATH))
        except RuntimeError as exc:
            raise RuntimeError(
                f"向量库被占用（{QDRANT_PATH}）。本地 Qdrant 同一时间只能被一个进程打开，"
                "重建索引前请先停掉 8090 问答服务。"
            ) from exc
    return _CLIENT


def _reset_store() -> None:
    """清空本地向量库目录。

    注意：qdrant 本地模式里 delete_collection + create_collection 并不会真的清掉旧点
    （实测重建后 count 仍保留旧数据），所以直接关闭客户端并删除存储目录。
    """
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:
            pass
        _CLIENT = None
    if QDRANT_PATH.exists():
        shutil.rmtree(QDRANT_PATH, ignore_errors=True)


def build() -> None:
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError("local_index.json 为空，请先运行 build_local_index.py")
    client = embedding_client()
    vectors = embed(client, [item["content"] for item in chunks[:1]])
    _reset_store()
    qdrant = qdrant_client()
    qdrant.create_collection(COLLECTION, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
    # 千问 embedding 单次最多 20 条，超出会报 InvalidParameter。
    batch_size = max(1, int(os.getenv("EMBEDDING_BATCH", "20")))
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        values = embed(client, [item["content"] for item in batch])
        qdrant.upsert(COLLECTION, [models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=item) for vector, item in zip(values, batch)])
    print(f"Qdrant 索引完成：{len(chunks)} 个分块 -> {QDRANT_PATH}")


def _doc_key(item: dict) -> str:
    """跨两路合并时用的去重键：同一条子块在两路里必须是同一个 key。"""
    metadata = item.get("metadata") or {}
    return f"{metadata.get('parent_id', '')}|{item.get('content', '')[:80]}"


def _rrf_retrieve(query: str, top_k: int) -> list[dict]:
    """向量 + BM25 各自独立召回，用 RRF（倒数排名融合）合并。"""
    pool = max(20, top_k * 4)
    rrf_k = float(os.getenv("RRF_K", "60"))
    vector_weight = float(os.getenv("RRF_VECTOR_WEIGHT", "0.5"))

    # 向量排名
    qdrant = qdrant_client()
    client = embedding_client()
    vector = embed(client, [query])[0]
    hits = qdrant.query_points(COLLECTION, query=vector, limit=pool, with_payload=True).points

    merged: dict[str, dict] = {}
    for rank, hit in enumerate(hits, 1):
        item = dict(hit.payload or {})
        item["score"] = item.get("score", 0.0)
        key = _doc_key(item)
        merged[key] = item
        merged[key]["_vec_rank"] = rank
        merged[key]["_vec_score"] = float(hit.score)

    # BM25 排名（独立召回，能带回向量漏掉的块）
    for rank, (_, item, raw) in enumerate(bm25_search(query, limit=pool), 1):
        key = _doc_key(item)
        entry = merged.setdefault(key, dict(item))
        entry["_bm_rank"] = rank
        entry["_bm_score"] = raw

    candidates = []
    for entry in merged.values():
        score_value = 0.0
        if entry.get("_vec_rank"):
            score_value += vector_weight / (rrf_k + entry["_vec_rank"])
        if entry.get("_bm_rank"):
            score_value += (1 - vector_weight) / (rrf_k + entry["_bm_rank"])
        entry["score"] = score_value
        candidates.append(entry)
    candidates.sort(key=lambda item: item["score"], reverse=True)

    shortlist = candidates[: max(top_k * 4, 20)]
    order = rerank(query, [item.get("content", "") for item in shortlist], top_n=top_k)
    if order:
        return [{**shortlist[index], "score": relevance} for index, relevance in order]
    return shortlist[:top_k]


def retrieve(query: str, top_k: int = 6) -> list[dict]:
    if os.getenv("FUSION", "weighted").lower() == "rrf":
        return _rrf_retrieve(query, top_k)

    qdrant = qdrant_client()
    client = embedding_client()
    vector = embed(client, [query])[0]
    hits = qdrant.query_points(COLLECTION, query=vector, limit=max(12, top_k * 3), with_payload=True).points
    candidates = []
    for hit in hits:
        item = dict(hit.payload or {})
        semantic = float(hit.score)
        lexical = score(query, item.get("content", ""))
        item["score"] = 0.7 * semantic + 0.3 * lexical
        candidates.append(item)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    shortlist = candidates[: max(top_k * 4, 20)]
    order = rerank(query, [item.get("content", "") for item in shortlist], top_n=top_k)
    if order:
        return [{**shortlist[index], "score": relevance} for index, relevance in order]
    return candidates[:top_k]


if __name__ == "__main__":
    build()
