"""混合检索：BM25 + 向量，RRF 融合。

嵌入支持两种后端：
- EMBED_BACKEND=ollama：调用本地 Ollama 的 /api/embed（默认，免下载模型）
- EMBED_BACKEND=local：用 sentence-transformers 加载本地模型（如 bge-m3）
"""
import json
import urllib.request
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    id: str
    text: str


def tokenize(text: str):
    """中文分词：优先 jieba，失败则退回按空白切分。"""
    try:
        import jieba
        return [t for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return [t for t in text.replace("\n", " ").split(" ") if t]


def ranks_from_scores(scores):
    """分数 -> 排名（降序，最高分排名 0）。"""
    order = np.argsort(-np.asarray(scores, dtype=float))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(order))
    return ranks


def rrf(rank_lists, k=60):
    """倒数排名融合（Reciprocal Rank Fusion）。"""
    n = len(rank_lists[0])
    score = np.zeros(n)
    for ranks in rank_lists:
        score += 1.0 / (k + ranks)
    return score


def load_docs(path: str, chunk_size: int = 200, overlap: int = 40):
    """读取文本文件，按段落/长度切分。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        step = max(1, chunk_size - overlap)
        for j in range(0, len(para), step):
            seg = para[j : j + chunk_size]
            if seg.strip():
                chunks.append(Chunk(id=f"{i}-{j}", text=seg))
    return chunks


class LocalEmbedder:
    """sentence-transformers 本地模型。"""

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        print(f"加载本地 embedding 模型：{model} ...")
        self.model = SentenceTransformer(model)

    def encode(self, texts, normalize_embeddings=True):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)


class OllamaEmbedder:
    """调用 Ollama /api/embed（无需额外下载模型）。"""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def encode(self, texts, normalize_embeddings=True):
        # 绕过系统代理（localhost 走代理会导致连接被重置）
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        vecs = []
        for t in texts:
            data = json.dumps({"model": self.model, "input": t}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/embed",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with opener.open(req, timeout=120) as resp:
                vecs.append(json.loads(resp.read())["embeddings"][0])
        arr = np.asarray(vecs, dtype=float)
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def build_embedder(backend: str, model: str):
    if backend == "ollama":
        print(f"使用 Ollama embedding 模型：{model}")
        return OllamaEmbedder(model)
    return LocalEmbedder(model)


class HybridRetriever:
    def __init__(self, chunks, embedder, top_k: int = 5):
        self.chunks = chunks
        self.top_k = top_k
        self.embedder = embedder
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.vectors = embedder.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str):
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
        q_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        vec_scores = self.vectors @ q_vec
        fused = rrf([ranks_from_scores(bm25_scores), ranks_from_scores(vec_scores)])
        top_idx = np.argsort(-fused)[: self.top_k]
        return [self.chunks[i] for i in top_idx]
