# 多 Agent 智能问答系统（检索 + 生成 + 校验 + 反思）

用 LangGraph 编排「检索 Agent → 生成 Agent → 校验 Agent」的多 Agent 协作流程，
通过校验反馈 + 反思重试降低 RAG 回答幻觉，并做「单 Agent vs 多 Agent」的忠实度对比评测。

## 架构

```
用户提问 → [检索 Agent] 改写 query + 混合检索(BM25+向量+RRF)
        → [生成 Agent] 基于上下文生成答案+引用
        → [校验 Agent] 检查忠实度与完整性
        → 通过 → 输出；不通过 → 带反馈回到生成（最多 2 轮反思）
```

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv venv
# Windows: venv\Scripts\activate   Linux/macOS: source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置模型
cp .env.example .env    # 编辑 .env 填入你的 LLM 端点

# 4. 命令行问答
python main.py

# 5. Web 前端（FastAPI + 流式输出）
uvicorn app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000

# 5. 对比评测（单 Agent vs 多 Agent）
python eval.py
```

## 模型配置说明

`.env` 里 `LLM_BASE_URL` 支持任何 OpenAI 兼容端点：

- **Ollama 本地**：先 `ollama pull qwen2.5:7b`，`.env` 用 `http://localhost:11434/v1`
- **vLLM / DeepSeek / OpenAI**：换成对应 base_url 和 key

`EMBED_MODEL` 首次运行会下载模型。想快速跑通可先用 `BAAI/bge-small-zh-v1.5`，
正式对比再换回 `BAAI/bge-m3`。

## Web 前端

`app.py` 提供 FastAPI 服务，`static/index.html` 是原生 JS 前端（无框架）：

- `GET /`：问答页面
- `POST /api/ask`：一次性问答（JSON）
- `POST /api/ask/stream`：流式问答（SSE，答案逐字输出）

前端会展示：答案（带引用）、检索查询、召回文档数、校验结果、反思重试轮数、参考来源。

## 项目结构

```
multi-agent-qa/
├── main.py           # LangGraph 多 Agent 图 + 命令行入口
├── app.py            # FastAPI Web 服务（含 SSE 流式端点）
├── static/index.html # 网页前端
├── agents.py         # 检索改写 / 生成 / 校验 三个 Agent 的 prompt
├── retrieval.py      # 手写 BM25 + 向量 + RRF 混合检索
├── eval.py           # LLM-as-judge 忠实度对比评测
├── .env.example      # 配置模板（.env 不入库）
├── config.py         # 环境变量配置
├── data/
│   ├── sample_docs.txt        # 示例知识库文档
│   └── eval_questions.json    # 评测问题集
└── requirements.txt
```

## 面试可讲的点

- 多 Agent 角色分工：检索、生成、校验各司其职
- 共享状态：LangGraph 的 State 在节点间传递
- 条件边路由：校验结果决定「结束」还是「反思重试」
- 结果校验与冲突裁决：校验 Agent 对生成结果做裁决
- 反思重试：critique 反馈喂回生成环节，最多 2 轮
- 混合检索：手写 BM25 + 向量 + RRF（能讲清原理）
