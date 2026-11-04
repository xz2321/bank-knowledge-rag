"""LangChain LCEL 版问答服务（运行在 .lc 环境，端口 8090）。"""
from __future__ import annotations

import os
import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from config import load_env

load_env()

import chat_store as store  # noqa: E402
import lc_rag  # noqa: E402
from retrieval import bm25_index  # noqa: E402
from build_local_index import find_pdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(_: FastAPI):
    bm25_index()
    store.init()
    print(
        f"[LC] 启动：FUSION=rrf(LCEL) | MIN_RELEVANCE={os.getenv('MIN_RELEVANCE', '0.35')} "
        f"| MODEL={os.getenv('CHAT_MODEL')}"
    )
    yield


app = FastAPI(title="银行知识库问答助手（LangChain LCEL）", version="2.0.0", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    question: str
    top_k: int = 6
    session_id: str | None = None
    history: list[ChatMessage] = []
    debug: bool = False


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (Path(__file__).with_name("qa.html")).read_text(encoding="utf-8")


@app.get("/source-pdf/{source_name}")
def source_pdf(source_name: str):
    filename = Path(source_name).name
    pdf = find_pdf(filename)
    if pdf is None or ROOT not in pdf.resolve().parents:
        raise HTTPException(404, "原始 PDF 不存在")
    return FileResponse(pdf, media_type="application/pdf", filename=pdf.name, content_disposition_type="inline")


@app.get("/sessions")
def list_sessions() -> dict:
    return {"sessions": store.list_sessions()}


@app.post("/sessions")
def create_session() -> dict:
    return {"session_id": store.create_session()}


@app.get("/sessions/{session_id}")
def session_detail(session_id: str) -> dict:
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": item["role"],
                "content": item["content"],
                "citations": item["citations"],
                "created_at": item["created_at"],
            }
            for item in store.get_messages(session_id)
        ],
    }


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str) -> dict:
    store.delete_session(session_id)
    return {"deleted": session_id}


@app.post("/reindex")
def reindex() -> dict:
    """重建向量库（由 8091 在知识库增删后调用）。"""
    from hybrid_retriever import COLLECTION, build, qdrant_client

    build()
    return {"points": qdrant_client().count(COLLECTION).count}


def _chat(request: ChatRequest, progress=None, on_token=None) -> dict:
    question = request.question.strip()
    session_id = request.session_id or store.create_session()
    stored = store.get_messages(session_id)
    if stored:
        history = [{"role": item["role"], "content": item["content"]} for item in stored][-6:]
    else:
        history = [{"role": item.role, "content": item.content} for item in request.history][-6:]

    result = lc_rag.answer(
        question, history=history, top_k=request.top_k, debug=request.debug,
        progress=progress, on_token=on_token,
    )

    store.add_message(session_id, "user", question)
    store.add_message(session_id, "assistant", result.get("answer", ""), result.get("citations", []))
    if not stored:
        store.rename_session(session_id, question)

    return {"session_id": session_id, **result}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    """兼容评测脚本和旧客户端的非流式接口。"""
    return _chat(request)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """边生成边传送草稿；完成轻量引用检查后发送最终结果。"""
    def encode(event: dict) -> str:
        return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    def events():
        messages: queue.Queue = queue.Queue()
        emitted_tokens = False

        def run_chat():
            try:
                result = _chat(
                    request,
                    lambda message: messages.put(("status", message)),
                    lambda token: messages.put(("token", token)),
                )
                messages.put(("result", result))
            except Exception as exc:
                messages.put(("error", str(exc)))

        threading.Thread(target=run_chat, daemon=True).start()
        yield encode({"type": "status", "message": "正在准备检索…"})
        while True:
            kind, value = messages.get()
            if kind == "status":
                yield encode({"type": "status", "message": value})
                continue
            if kind == "token":
                emitted_tokens = True
                yield encode({"type": "token", "content": value})
                continue
            if kind == "error":
                yield encode({"type": "error", "message": value})
                break

            result = value
            answer = result.get("answer", "")
            if not emitted_tokens:
                for offset in range(0, len(answer), 18):
                    yield encode({"type": "token", "content": answer[offset:offset + 18]})
            yield encode({"type": "done", "result": result})
            break

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lc_qa_app:app", host="127.0.0.1", port=int(os.getenv("QA_PORT", "8090")), reload=False)
