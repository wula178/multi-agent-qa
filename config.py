"""配置加载：从环境变量读取模型与检索参数"""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "ollama")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_RETRY_ROUNDS = int(os.getenv("MAX_RETRY_ROUNDS", "2"))
