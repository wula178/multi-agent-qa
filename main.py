"""多 Agent 问答系统入口：LangGraph 编排「检索 -> 生成 -> 校验 -> 反思重试」。"""
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

import config
from agents import check_answer, generate_answer, get_llm, rewrite_queries
from retrieval import Chunk, HybridRetriever, build_embedder, load_docs


class AgentState(TypedDict):
    question: str
    queries: List[str]
    docs: List[Chunk]
    answer: str
    critique: str
    passed: bool
    round: int


def build_graph(retriever: HybridRetriever):
    llm = get_llm()

    # ① 检索 Agent：改写 query + 多查询混合检索
    def retrieve(state):
        queries = rewrite_queries(llm, state["question"])
        docs, seen = [], set()
        for q in queries:
            for c in retriever.search(q):
                if c.id not in seen:
                    seen.add(c.id)
                    docs.append(c)
        return {"queries": queries, "docs": docs, "round": 0}

    # ② 生成 Agent：基于上下文生成答案（带上一轮校验反馈）
    def generate(state):
        answer = generate_answer(
            llm, state["question"], state["docs"], state.get("critique", "")
        )
        return {"answer": answer, "round": state.get("round", 0) + 1}

    # ③ 校验 Agent：裁决是否通过
    def critique(state):
        text, passed = check_answer(
            llm, state["question"], state["answer"], state["docs"]
        )
        return {"critique": text, "passed": passed}

    # 条件路由：通过->结束；未通过且未超轮次->反思重试；否则兜底结束
    def router(state):
        if state["passed"]:
            return "end"
        if state["round"] < config.MAX_RETRY_ROUNDS:
            return "retry"
        return "end"

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("critique", critique)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "critique")
    g.add_conditional_edges("critique", router, {"retry": "generate", "end": END})
    return g.compile()


def main():
    chunks = load_docs("data/sample_docs.txt")
    embedder = build_embedder(config.EMBED_BACKEND, config.EMBED_MODEL)
    retriever = HybridRetriever(chunks, embedder, config.TOP_K)
    graph = build_graph(retriever)

    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("请输入问题：").strip()
    result = graph.invoke({"question": question})

    print("\n" + "=" * 60)
    print("问题：", result["question"])
    print("检索查询：", " | ".join(result["queries"]))
    print("召回文档数：", len(result["docs"]))
    print("-" * 60)
    print("答案：")
    print(result["answer"])
    print("-" * 60)
    print("校验结果：", result["critique"])
    print("反思重试轮次：", max(0, result["round"] - 1))


if __name__ == "__main__":
    main()
