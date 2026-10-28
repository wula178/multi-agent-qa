"""FastAPI 前端：把多 Agent 问答系统暴露为 Web 接口。

- GET  /            网页
- POST /api/ask     一次性问答（返回 JSON）
- POST /api/ask/stream  流式问答（SSE，逐 token 输出）

启动：uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import config
from agents import check_answer, get_llm, rewrite_queries, stream_generate
from retrieval import HybridRetriever, build_embedder, load_docs

# 懒加载：第一次请求时才加载 embedding 模型和构建图
_state = {}


def _init():
    if "retriever" not in _state:
        print("初始化：加载文档 + embedding 模型 ...")
        chunks = load_docs("data/sample_docs.txt")
        embedder = build_embedder(config.EMBED_BACKEND, config.EMBED_MODEL)
        _state["retriever"] = HybridRetriever(chunks, embedder, config.TOP_K)
        print("初始化完成。")
    return _state["retriever"]


def _retrieve(llm, question):
    queries = rewrite_queries(llm, question)
    retriever = _init()
    docs, seen = [], set()
    for q in queries:
        for c in retriever.search(q):
            if c.id not in seen:
                seen.add(c.id)
                docs.append(c)
    return queries, docs


app = FastAPI(title="多 Agent 智能问答系统")


class AskRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index():
    return Path("static/index.html").read_text(encoding="utf-8")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/ask")
def ask(req: AskRequest):
    llm = get_llm()
    queries, docs = _retrieve(llm, req.question)

    # 生成 + 校验 + 反思重试（非流式）
    critique, answer, passed, attempt = "", "", False, 0
    while True:
        from agents import generate_answer
        answer = generate_answer(llm, req.question, docs, critique)
        critique, passed = check_answer(llm, req.question, answer, docs)
        attempt += 1
        if passed or attempt >= config.MAX_RETRY_ROUNDS:
            break

    return {
        "question": req.question,
        "queries": queries,
        "answer": answer,
        "critique": critique,
        "passed": passed,
        "retry_rounds": attempt - 1,
        "doc_count": len(docs),
        "sources": [{"id": c.id, "text": c.text[:120]} for c in docs],
    }


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    llm = get_llm()
    queries, docs = _retrieve(llm, req.question)

    def generate():
        yield _sse({"type": "start", "queries": queries, "doc_count": len(docs)})

        critique, answer, passed, attempt = "", "", False, 0
        while True:
            if attempt > 0:
                yield _sse({"type": "retry", "critique": critique})
            answer = ""
            for token in stream_generate(llm, req.question, docs, critique):
                answer += token
                yield _sse({"type": "token", "content": token})
            critique, passed = check_answer(llm, req.question, answer, docs)
            attempt += 1
            if passed or attempt >= config.MAX_RETRY_ROUNDS:
                break

        yield _sse({
            "type": "done",
            "answer": answer,
            "passed": passed,
            "critique": critique,
            "retry_rounds": attempt - 1,
            "sources": [{"id": c.id, "text": c.text[:120]} for c in docs],
        })

    return StreamingResponse(generate(), media_type="text/event-stream")
