"""三个 Agent 的 prompt 与实现：检索改写 / 生成 / 校验。

LLM 调用直接用 httpx 走 Ollama 原生接口，绕过 langchain-openai 的传输层
（它在有系统代理的 Windows 上容易卡住/走代理）。
"""
import json

import httpx

import config


class SimpleResp:
    def __init__(self, content: str):
        self.content = content


class SimpleLLM:
    def __init__(self):
        # trust_env=False：不读系统/环境代理，保证 localhost 的 Ollama 直连
        self.client = httpx.Client(
            base_url=config.LLM_BASE_URL,
            timeout=httpx.Timeout(connect=15, read=300, write=30, pool=15),
            trust_env=False,
        )

    def invoke(self, prompt: str, max_tokens: int = 1024):
        # 用 Ollama 原生 /api/chat，think:false 关闭思考（否则 qwen3.5 只输出 reasoning）
        resp = self.client.post(
            "/api/chat",
            json={
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        return SimpleResp(resp.json()["message"]["content"])

    def stream(self, prompt: str, max_tokens: int = 1024):
        """流式生成：逐段 yield 内容（Ollama NDJSON 流）。"""
        with self.client.stream(
            "POST",
            "/api/chat",
            json={
                "model": config.LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "think": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    break
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content


def get_llm(temperature: float = 0):
    return SimpleLLM()


RETRIEVE_PROMPT = """你是一个检索查询改写器。把用户问题改写成 2-3 个适合做关键词+向量检索的查询。
只输出查询，每行一个，不要解释、不要编号。

用户问题：{question}
"""

GENERATE_PROMPT = """根据下面的资料回答问题。要求：
1. 只能依据资料回答，资料里没有的内容说"资料中未提供"；
2. 每条结论后标注来源编号，如 [1][2]；
3. 不要编造、不要补充资料之外的知识。

资料：
{context}

问题：{question}

{critique_hint}"""

CRITIQUE_PROMPT = """你是答案校验器。检查答案是否同时满足：
1. 忠实：答案的每句话都能被资料支撑，没有编造；
2. 完整：回答了用户的问题。

满足就输出 PASS，否则输出 FAIL 并说明具体问题。
只输出 PASS 或 FAIL，加一句理由，不要其他内容。

资料：
{context}

问题：{question}

答案：
{answer}
"""


def rewrite_queries(llm, question: str):
    resp = llm.invoke(RETRIEVE_PROMPT.format(question=question), max_tokens=200)
    lines = [
        l.strip().lstrip("-•* ").strip()
        for l in resp.content.strip().splitlines()
        if l.strip()
    ]
    return lines[:3]


def _build_context(chunks):
    return "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))


def _build_hint(critique):
    return (
        f"上一轮校验反馈：{critique}\n请针对反馈修正后重新回答。"
        if critique
        else ""
    )


def generate_answer(llm, question: str, chunks, critique: str = ""):
    prompt = GENERATE_PROMPT.format(
        question=question, context=_build_context(chunks), critique_hint=_build_hint(critique)
    )
    resp = llm.invoke(prompt, max_tokens=1024)
    return resp.content


def stream_generate(llm, question: str, chunks, critique: str = ""):
    """流式生成答案，逐段 yield 文本。"""
    prompt = GENERATE_PROMPT.format(
        question=question, context=_build_context(chunks), critique_hint=_build_hint(critique)
    )
    yield from llm.stream(prompt, max_tokens=1024)


def check_answer(llm, question: str, answer: str, chunks):
    context = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    resp = llm.invoke(
        CRITIQUE_PROMPT.format(question=question, answer=answer, context=context),
        max_tokens=300,
    )
    text = resp.content.strip()
    return text, text.upper().startswith("PASS")
