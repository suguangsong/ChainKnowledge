# ChainKnowledge

> 一个可直接运行的企业内部知识库问答案例，覆盖 LangChain 核心能力，并具备企业落地能力。

## 一、项目目标

`ChainKnowledge` 以企业内部文档为知识源（产品文档、员工手册、技术文档、运营 SOP），构建：

- 多格式文档上传
- 文本分块与向量化
- 检索增强生成（RAG）
- 检索重排
- 对话记忆
- 回答溯源
- API 化与前后端分离

## 二、技术栈

- **LLM/Embedding**：OpenAI（默认）或 ChatGLM（兼容 OpenAI 协议）
- **向量数据库**：Chroma
- **应用框架**：LangChain + Streamlit + FastAPI
- **检索重排**：向量分数 + 关键词重叠（可配置权重）
- **部署**：Docker + Nginx + Compose（含反向代理与持久化挂载）
- **文档能力**：PDF、DOCX、TXT、MD、HTML、CSV

## 三、目录结构（分层设计）

```text
src/chainknowledge/
  __init__.py
  core/
    __init__.py
    config.py                  # 运行时配置与路径
    llm.py                     # LLM 与 Embedding 工厂
    loader.py                  # 多格式文档加载
    splitter.py                # 文本分割策略
    vector_store.py            # Chroma 向量库管理
    memory.py                  # 对话记忆
    reranker.py                # 检索重排
  services/
    __init__.py
    ingestion.py               # 文档入库服务
    qa_service.py              # RAG 问答与重排管道
  ui/
    __init__.py
    streamlit_app.py           # Streamlit 演示界面
  api/
    __init__.py
    main.py                    # FastAPI API 服务

frontend/
  index.html                  # 简易前端示例

docker/
  nginx.conf                  # Nginx 反代配置

Dockerfile

docker-compose.yml
README.md
.env.example
requirements.txt
BLOG.md
```

## 四、快速开始

### 1) 安装依赖

```bash
python -m venv .venv
. ./.venv/Scripts/Activate.ps1   # PowerShell
pip install -r requirements.txt
```

### 2) 配置环境变量

```bash
cp .env.example .env
```

填入 `OPENAI_API_KEY` 或 `CHATGLM_API_KEY`。如需 Streamlit 与 API 分别使用不同凭据，可继续设置

`API_OPENAI_API_KEY`、`API_CHATGLM_API_KEY`、`STREAMLIT_OPENAI_API_KEY`、`STREAMLIT_CHATGLM_API_KEY`。

### 3) 本地启动 API（推荐）

```bash
uvicorn chainknowledge.api.main:app --reload --host 0.0.0.0 --port 8000
```

- 接口文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/health`

### 4) 本地启动 Streamlit 纯前端演示（可选）

```bash
streamlit run src/chainknowledge/ui/streamlit_app.py
```

### 5) 本地静态前端 + API 联合验证（可选）

- 直接在浏览器打开 `frontend/index.html`（需先启动 API，且静态服务可将 `/api/*` 代理到 `127.0.0.1:8000`）。

## 五、Docker + Nginx 启动

### 1) 准备镜像与服务

```bash
docker compose up --build -d
```

### 2) 访问

- 前端：`http://127.0.0.1:8080`
- 后端 API：`http://127.0.0.1:8080/api/health`

## 六、配置说明（重点）

`RETRIEVER_TOP_K`、`RERANK_CANDIDATE_MULTIPLIER`、`RERANK_TOP_K`、`RERANK_ALPHA` 分别控制：

- 向量首次返回数量
- 重排候选放大倍率
- 重排后保留数量
- 重排中的向量/关键词权重比例

`RERANK_ENABLED=false` 可关闭重排，直接使用向量检索 TopK 的结果。

`API_OPENAI_API_KEY` / `API_CHATGLM_API_KEY` 用于 API 服务，`STREAMLIT_*` 用于 Streamlit；任一未设置则回退到对应通用 key。

`/api/chat` 响应已返回检索统计：

- `candidate_k`：本次查询计划检索的候选上限
- `retrieved_count`：向量检索实际返回条目数
- `returned_count`：重排后进入 LLM 的上下文条目数

## 七、校验与自检

在本地可做三类快速校验：

1. Python 语法检查

```bash
python -m compileall -q src
```

2. 依赖可用性检查

```bash
python -m pip check
```

3. API 路由与状态检查

```bash
uvicorn chainknowledge.api.main:app --host 127.0.0.1 --port 8000
```

然后请求：

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/status
```

4. 直接调用 /api/chat（可执行示例）

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","question":"请说明公司请假流程","provider":"openai","top_k":4,"reranker_candidate_multiplier":3,"reranker_top_k":4,"reranker_alpha":0.7,"reranker_enabled":true,"memory_window":6}'
```

成功返回示例字段（已包含重排效果可视化指标）：

```json
{
  "session_id": "demo",
  "answer": "..."
}
```

你可以重点关注 `candidate_k`、`retrieved_count`、`returned_count` 这三个字段用于对比重排前后的检索效果。

## 八、常见扩展

- 替换为企业真实 LLM 接口（OpenAI/ChatGLM）和私有向量库（Milvus/pgvector/Pinecone）
- 接入权限体系后端和审计日志
- 加入多租户 collection 隔离
- 将重排器升级为 CrossEncoder 等深度模型
