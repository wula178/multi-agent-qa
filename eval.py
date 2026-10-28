"""对比评测：单次生成 vs 多 Agent（校验 + 反思），用 LLM-as-judge 打分忠实度。"""
import json

import config
from agents import generate_answer, get_llm
from main import build_graph
from retrieval import HybridRetriever, build_embedder, load_docs

JUDGE_PROMPT = """你是评测裁判。判断"答案"是否忠实于"资料"（没有编造资料之外的内容）。
只输出一个 0 到 1 之间的小数（如 0.8），表示忠实度，不要解释。

资料：
{context}

问题：
{question}

答案：
{answer}
"""


def judge_faithfulness(llm, question, answer, docs):
    context = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(docs))
    resp = llm.invoke(
        JUDGE_PROMPT.format(question=question, answer=answer, context=context)
    )
    try:
        return float(resp.content.strip())
    except ValueError:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def main():
    questions = json.load(open("data/eval_questions.json", encoding="utf-8"))
    chunks = load_docs("data/sample_docs.txt")
    embedder = build_embedder(config.EMBED_BACKEND, config.EMBED_MODEL)
    retriever = HybridRetriever(chunks, embedder, config.TOP_K)
    llm = get_llm()
    graph = build_graph(retriever)

    single_scores, multi_scores = [], []
    for q in questions:
        docs = retriever.search(q)
        single = generate_answer(llm, q, docs)
        s = judge_faithfulness(llm, q, single, docs)

        result = graph.invoke({"question": q})
        m = judge_faithfulness(llm, q, result["answer"], result["docs"])

        single_scores.append(s)
        multi_scores.append(m)
        print(f"Q: {q}")
        print(f"  单次生成忠实度={s}   多Agent忠实度={m}   校验={result['passed']}")

    print("\n" + "=" * 60)
    print(f"平均忠实度：单次生成={mean(single_scores):.3f}   多Agent={mean(multi_scores):.3f}")


if __name__ == "__main__":
    main()
