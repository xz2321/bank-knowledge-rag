"""LangChain LCEL 版 RAG 链路。

链路：多轮改写（LCEL） → 双路检索（LangChain Retriever） → RRF 融合 → 千问重排（Runnable）
      → 阈值/证据判断（RunnableBranch） → 提示词（ChatPromptTemplate） → DeepSeek（ChatOpenAI）
      → 引用校验。

说明：
- 向量召回失败（如账号欠费/网络问题）时自动降级为 BM25，链路不中断；
- 重排失败时保留融合顺序；
- 拒答、引用编号校验复用 qa_assistant 里已验证的实现，保证行为一致。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import load_env

load_env()

from retrieval import bm25_search, load_chunks, saturate  # noqa: E402
from rerank import rerank  # noqa: E402
from qa_assistant import (  # noqa: E402
    ADVICE_RE,
    build_citations,
    chat_client,
    format_answer_markdown,
    has_evidence,
    small_talk_reply,
    source_path_for,
    validate_citations,
)

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(ROOT / "knowledge_base_output" / "qdrant")))
COLLECTION = os.getenv("QDRANT_COLLECTION", "bank_knowledge")

SYSTEM_PROMPT = (
    "你是严谨的银行考试知识库助手。只根据给定资料回答，不要补充资料以外的知识。"
    "每条结论后面用 [编号] 标注依据，编号必须来自资料；引用内容必须直接支持对应结论，不能只因主题相近就引用。"
    "资料不足时明确回答“资料中没有找到依据”，不要编造。"
    "同一概念只解释一次，按问题直接组织答案，不要把不同来源的重复定义堆叠起来。"
    "回答简明，重点通常不超过五条。列表统一使用连续的 1. 2. 3. 格式；公式写成可读的普通文本，不要输出 LaTeX 分隔符或转义反斜杠。"
    "当用户问“怎么理解”“是什么意思”等概念题，且概念涉及公式、比例或变化关系时，优先按“直觉解释—必要公式—一个简单例子—关键分类或例外”组织，最多四点。"
    "例子可用简单假设数字演示公式，但要明确写“假设举例”，并确保只是根据给定定义做的计算，不可把假设说成资料事实。"
    "公式只保留理解所需的简式，并用一句话解释变量或正负号；不要罗列多个等价推导式。"
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "根据对话把用户最后一个问题改写成一个可以单独检索的完整问句。"
            "必须写出上文讨论的具体主题，不能出现“它、这个、上述、该”等指代词；"
            "只输出这一句问句，不要解释。",
        ),
        ("human", "对话：\n{history}\n\n最后一个问题：{question}\n改写后的问句："),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "资料：\n{context}\n\n问题：{question}"),
    ]
)


def chat_llm(temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=temperature,
    )


class QwenEmbeddings(Embeddings):
    """千问 OpenAI 兼容 embedding（1024 维）。"""

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        data = self.client.embeddings.create(model=self.model, input=texts).data
        return [item.embedding for item in data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class QdrantRetriever(BaseRetriever):
    """向量召回；失败返回空列表，由融合器降级到 BM25。"""

    embeddings: Embeddings
    k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        from hybrid_retriever import qdrant_client

        try:
            vector = self.embeddings.embed_query(query)
            hits = qdrant_client().query_points(
                COLLECTION, query=vector, limit=self.k, with_payload=True
            ).points
        except Exception as exc:
            print(f"[LC] 向量召回不可用，降级 BM25：{exc}"[:200])
            return []
        documents: list[Document] = []
        for hit in hits:
            payload = dict(hit.payload or {})
            metadata = dict(payload.get("metadata") or {})
            metadata.update(
                {
                    "title": payload.get("title", ""),
                    "vector_score": float(hit.score),
                    "score": float(hit.score),
                }
            )
            documents.append(Document(page_content=payload.get("content", ""), metadata=metadata))
        return documents


class Bm25Retriever(BaseRetriever):
    """BM25 独立召回；与向量召回去重合并。"""

    k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        documents: list[Document] = []
        for _, item, raw in bm25_search(query, limit=self.k):
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "title": item.get("title", ""),
                    "bm25_raw": float(raw),
                    "bm25_score": saturate(float(raw)),
                    "score": saturate(float(raw)),
                }
            )
            documents.append(Document(page_content=item.get("content", ""), metadata=metadata))
        return documents


class FusionRetriever(BaseRetriever):
    """双路 RRF（倒数排名融合），向量路失败时自动退化为纯 BM25。"""

    vector: BaseRetriever
    bm25: BaseRetriever
    vector_weight: float = 0.5
    c: int = 60
    k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        merged: dict[str, Document] = {}
        ranks: dict[str, dict[str, int]] = {}

        def key_of(doc: Document) -> str:
            parent = doc.metadata.get("parent_id") or doc.metadata.get("title", "")
            return f"{parent}|{doc.page_content[:80]}"

        for rank, doc in enumerate(self.vector.invoke(query), 1):
            key = key_of(doc)
            merged[key] = doc
            ranks.setdefault(key, {})["vector"] = rank

        for rank, doc in enumerate(self.bm25.invoke(query), 1):
            key = key_of(doc)
            merged.setdefault(key, doc)
            ranks.setdefault(key, {})["bm25"] = rank

        scored: list[tuple[float, Document]] = []
        for key, doc in merged.items():
            entry = ranks[key]
            rrf = 0.0
            vector_rank = entry.get("vector")
            bm25_rank = entry.get("bm25")
            if vector_rank:
                rrf += self.vector_weight / (self.c + vector_rank)
            if bm25_rank:
                rrf += (1 - self.vector_weight) / (self.c + bm25_rank)
            metadata = dict(doc.metadata)
            # 融合分只用于排序；对外分数仍用 0~1 的可比分数（向量余弦 / BM25 饱和分），
            # 这样阈值 0.35 和证据判断的语义不会被 RRF 的小数值破坏。
            vector_score = float(metadata.get("vector_score", 0.0) or 0.0)
            bm25_score = float(metadata.get("bm25_score", 0.0) or 0.0)
            if vector_score and bm25_score:
                comparable = 0.7 * vector_score + 0.3 * bm25_score
            else:
                comparable = vector_score or bm25_score
            metadata["fusion_score"] = rrf
            metadata["score"] = comparable
            scored.append((rrf, Document(page_content=doc.page_content, metadata=metadata)))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[: self.k]]


def rerank_documents(payload: dict) -> dict:
    """Runnable：把融合候选交给千问重排，失败保留原顺序。"""
    query = payload["query"]
    documents: list[Document] = payload["documents"]
    if not documents:
        return payload
    order = rerank(query, [doc.page_content for doc in documents], top_n=len(documents))
    if not order:
        return payload
    reranked = []
    for index, relevance in order:
        metadata = dict(documents[index].metadata)
        metadata["relevance"] = float(relevance)
        reranked.append(Document(page_content=documents[index].page_content, metadata=metadata))
    return {**payload, "documents": reranked}


def records_from_documents(documents: list[Document]) -> list[dict]:
    records = []
    for document in documents:
        metadata = dict(document.metadata)
        title = metadata.pop("title", "")
        score = float(metadata.pop("relevance", metadata.pop("score", 0.0)))
        records.append(
            {
                "title": title,
                "content": document.page_content,
                "score": score,
                "metadata": metadata,
            }
        )
    return records


def recall_depth() -> int:
    """两路召回的深度：默认取全部子块，可用 RECALL_DEPTH 覆盖。"""
    value = os.getenv("RECALL_DEPTH", "").strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    return max(1, len(load_chunks()))


class EmptyRetriever(BaseRetriever):
    """关闭向量检索时用的空召回器（纯 BM25 模式）。"""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return []


def build_retriever(top_k: int) -> FusionRetriever:
    depth = recall_depth()
    use_vector = os.getenv("USE_QDRANT", "1") == "1"
    vector: BaseRetriever = QdrantRetriever(embeddings=QwenEmbeddings(), k=depth) if use_vector else EmptyRetriever()
    return FusionRetriever(
        vector=vector,
        bm25=Bm25Retriever(k=depth),
        vector_weight=float(os.getenv("RRF_VECTOR_WEIGHT", "0.5")),
        c=int(os.getenv("RRF_K", "60")),
        k=max(20, top_k * 4),
    )


# ---- 两个 LCEL 链：改写、生成 ----
rewrite_chain = REWRITE_PROMPT | chat_llm(temperature=0) | StrOutputParser()
answer_chain = ANSWER_PROMPT | chat_llm(temperature=0.2) | StrOutputParser()


def condense(history: list[dict], question: str) -> str:
    if not history:
        return question
    dialogue = "\n".join(
        f"{'用户' if item.get('role') == 'user' else '助手'}：{item.get('content', '')}" for item in history[-4:]
    )
    try:
        rewritten = rewrite_chain.invoke({"history": dialogue, "question": question}).strip()
        rewritten = re.sub(r"^(改写[:：]?)", "", rewritten).strip().strip('"“”').splitlines()[0]
        return rewritten or question
    except Exception as exc:
        print(f"[LC] 改写失败：{exc}"[:160])
        return question


def format_context(citations: list[dict]) -> str:
    return "\n\n".join(
        f"[{item['id']}] 来源：{item['title']}（{item['heading'] or '正文'}）\n{item['text']}" for item in citations
    )


def answer(
    question: str,
    history: list[dict] | None = None,
    top_k: int = 6,
    debug: bool = False,
    progress=None,
    on_token=None,
) -> dict:
    """执行一次 LCEL 问答，返回与旧接口一致的字段。"""
    history = history or []
    question = (question or "").strip()

    # 1) 寒暄/选股等意图分流（不检索）
    small_talk = small_talk_reply(question)
    if small_talk:
        return {"mode": "chat", "answer": small_talk, "citations": [], "sources": [], "grounded": True,
                "best_score": 0.0, "question_used": question}
    if ADVICE_RE.search(question):
        return {"mode": "refuse", "answer": "本知识库只覆盖银行考试资料，不提供选股、买卖或投资建议。",
                "citations": [], "sources": [], "grounded": False, "best_score": 0.0,
                "question_used": question, "reason": "policy"}

    # 2) LCEL：改写 → 检索（FusionRetriever）→ 重排（Runnable）
    if progress:
        progress("正在检索相关资料…")
    search_query = condense(history, question)
    retriever = build_retriever(top_k)
    retrieval = RunnablePassthrough.assign(
        documents=RunnableLambda(lambda payload: retriever.invoke(payload["query"]))
    ) | RunnableLambda(rerank_documents)
    retrieved = retrieval.invoke({"query": search_query})
    documents: list[Document] = retrieved["documents"]
    # 重排覆盖整个候选池，但对外只保留前 top_k，与旧版/评测口径保持一致
    records = records_from_documents(documents[: max(1, top_k)])

    threshold = float(os.getenv("MIN_RELEVANCE", "0.35"))
    best = max((float(record.get("score", 0) or 0) for record in records), default=0.0)
    weak = bool(records) and best < float(os.getenv("WEAK_EVIDENCE_SCORE", "0.85")) and not has_evidence(records, search_query)
    if not records or best < threshold or weak:
        if progress:
            progress("没有找到足够相关的依据，正在整理答复…")
        return {
            "mode": "refuse",
            "answer": "知识库中没有找到依据。",
            "citations": [],
            "sources": [],
            "grounded": False,
            "best_score": round(best, 4),
            "threshold": threshold,
            "question_used": search_query,
            "reason": "weak_evidence" if weak else "low_score",
        }

    # 3) 引用 + 校验
    if progress:
        progress("已找到相关资料，正在生成答案…")
    citations = build_citations(records, search_query)
    # 4) LCEL 生成链
    messages = [
        {"role": item.get("role", "user"), "content": item.get("content", "")}
        for item in history[-4:]
        if item.get("role") in {"user", "assistant"}
    ]
    chain_input = {"context": format_context(citations), "question": question, "history": messages}
    if on_token:
        chunks = []
        for chunk in answer_chain.stream(chain_input):
            if chunk:
                chunks.append(chunk)
                on_token(chunk)
        text = "".join(chunks)
    else:
        text = answer_chain.invoke(chain_input)
    if progress:
        progress("答案已生成，正在检查引用编号和主张覆盖…")
    text = format_answer_markdown(text)
    text, citations, citation_check = validate_citations(text, citations)
    citation_check["support_review"] = "disabled"
    if progress:
        progress("引用检查完成，正在发送答案…")

    public_citations = [
        {
            "id": item["id"],
            "title": item["title"],
            "heading": item["heading"],
            "score": item["score"],
            "snippet": item["snippet"],
            "source_path": item.get("source_path") or source_path_for(item["title"]),
            "parent_id": item.get("parent_id", ""),
            "page": item.get("page"),
            "page_end": item.get("page_end"),
            "page_estimated": item.get("page_estimated", False),
            "pdf_url": item.get("pdf_url", ""),
        }
        for item in citations
    ]
    if debug:
        # 评测需要判断“被引用的整节里有没有考点”，snippet 只有 150 字可能截掉，
        # 所以只在 debug 时附带父块全文，避免正常响应变大。
        for public, item in zip(public_citations, citations):
            public["context"] = item.get("text", "")
    payload = {
        "mode": "answer",
        "answer": text,
        "citations": public_citations,
        "sources": [{"title": item["title"], "score": item["score"]} for item in citations],
        "grounded": True,
        "best_score": round(best, 4),
        "threshold": threshold,
        "question_used": search_query,
        "citation_check": citation_check,
    }
    if debug:
        payload["retrieved"] = [
            {
                "title": record.get("title"),
                "heading": (record.get("metadata") or {}).get("heading", ""),
                "score": round(float(record.get("score", 0) or 0), 4),
                "content": record.get("content", ""),
            }
            for record in records
        ]
    return payload
