from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chainknowledge.core.config import AppConfig
from chainknowledge.core.llm import create_chat_model, create_embedding_model
from chainknowledge.core.memory import create_memory
from chainknowledge.core.vector_store import clear_vector_store, create_or_load_vector_store, get_vector_count
from chainknowledge.services.ingestion import ingest_uploads
from chainknowledge.services.qa_service import build_qa_chain, answer_with_sources

load_dotenv()

app = FastAPI(title="ChainKnowledge API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_CONFIG: Optional[AppConfig] = None
EMBEDDING_MODEL = None
VECTOR_STORE = None
LLM_CACHE = {}
MEMORY_CACHE: dict[str, Tuple[int, object]] = {}


class SourceItem(BaseModel):
    idx: int
    file: str
    page: str | int | None = None
    snippet: str


class QueryRequest(BaseModel):
    session_id: str = Field(default="default")
    question: str = Field(min_length=1, max_length=3000)
    provider: str = Field(default="openai")
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    reranker_top_k: Optional[int] = Field(default=None, ge=1, le=20)
    reranker_candidate_multiplier: Optional[int] = Field(default=None, ge=1, le=20)
    reranker_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reranker_enabled: Optional[bool] = None
    memory_window: Optional[int] = Field(default=None, ge=1, le=60)


class QueryResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceItem]
    top_k: int
    candidate_k: int
    retrieved_count: int
    returned_count: int
    reranker_enabled: bool


class UploadResponse(BaseModel):
    file_count: int
    chunk_count: int
    failed: List[str]


class StatusResponse(BaseModel):
    vector_count: int
    collection_name: str
    configured_provider: str
    ready: bool = True


class ApiMessage(BaseModel):
    message: str


class _UploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    @property
    def size(self) -> int:
        return len(self._content)

    def getvalue(self) -> bytes:
        return self._content


def _clamp_int(value: Optional[int], default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    return max(minimum, min(maximum, int(value)))


def _clamp_float(value: Optional[float], default: float, minimum: float, maximum: float) -> float:
    if value is None:
        return default
    return max(minimum, min(maximum, float(value)))


def _bootstrap() -> tuple[AppConfig, object, object]:
    global BASE_CONFIG, EMBEDDING_MODEL, VECTOR_STORE

    if BASE_CONFIG is not None and EMBEDDING_MODEL is not None and VECTOR_STORE is not None:
        return BASE_CONFIG, EMBEDDING_MODEL, VECTOR_STORE

    config = AppConfig.from_env()
    config.ensure_paths()
    embedding_model = create_embedding_model(
        config,
        api_key_override=config.openai_api_key,
        base_url_override=config.openai_api_base,
    )
    vector_store = create_or_load_vector_store(embedding_model, config)

    BASE_CONFIG = config
    EMBEDDING_MODEL = embedding_model
    VECTOR_STORE = vector_store
    return BASE_CONFIG, EMBEDDING_MODEL, VECTOR_STORE


def _normalize_provider(provider: str) -> str:
    normalized = (provider or "openai").lower()
    if normalized not in {"openai", "chatglm"}:
        raise HTTPException(status_code=400, detail="provider 仅支持 openai 或 chatglm")
    return normalized


def _build_runtime(request: QueryRequest, base: AppConfig) -> tuple[AppConfig, str]:
    provider = _normalize_provider(request.provider)
    runtime_api_key = (request.api_key or "").strip()

    if provider == "openai":
        model = (request.model or base.openai_model).strip() or base.openai_model
        runtime_api_key = runtime_api_key or base.api_openai_api_key
        api_base = (request.api_base if request.api_base is not None else base.openai_api_base).strip()
        runtime = replace(
            base,
            llm_provider="openai",
            openai_model=model,
            openai_api_base=api_base,
        )
    else:
        model = (request.model or base.chatglm_model).strip() or base.chatglm_model
        runtime_api_key = runtime_api_key or base.api_chatglm_api_key
        api_base = (request.api_base if request.api_base is not None else base.chatglm_api_base).strip()
        runtime = replace(
            base,
            llm_provider="chatglm",
            chatglm_model=model,
            chatglm_api_base=api_base,
        )

    if not runtime_api_key:
        raise HTTPException(status_code=400, detail="请先提供对应供应商的 API Key")

    runtime = replace(
        runtime,
        retriever_top_k=_clamp_int(request.top_k, base.retriever_top_k, 1, 20),
        reranker_top_k=_clamp_int(
            request.reranker_top_k,
            base.reranker_top_k,
            1,
            50,
        ),
        reranker_candidate_multiplier=_clamp_int(
            request.reranker_candidate_multiplier,
            base.reranker_candidate_multiplier,
            1,
            10,
        ),
        reranker_alpha=_clamp_float(request.reranker_alpha, base.reranker_alpha, 0.0, 1.0),
        reranker_enabled=base.reranker_enabled if request.reranker_enabled is None else request.reranker_enabled,
        memory_window=_clamp_int(request.memory_window, base.memory_window, 1, 60),
    )

    return runtime, runtime_api_key


def _get_llm(config: AppConfig, runtime_api_key: str):
    model = config.openai_model if config.llm_provider == "openai" else config.chatglm_model
    base_url = config.openai_api_base if config.llm_provider == "openai" else config.chatglm_api_base
    cache_key = (config.llm_provider, model, base_url, runtime_api_key)

    llm = LLM_CACHE.get(cache_key)
    if llm is not None:
        return llm

    llm = create_chat_model(
        config,
        provider=config.llm_provider,
        api_key_override=runtime_api_key,
        model_override=model,
        base_url_override=base_url,
    )
    LLM_CACHE[cache_key] = llm
    return llm


def _get_memory(session_id: str, window: int):
    cached = MEMORY_CACHE.get(session_id)
    if cached and cached[0] == window:
        return cached[1]

    memory = create_memory(window)
    MEMORY_CACHE[session_id] = (window, memory)
    return memory


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    try:
        config, _, vector_store = _bootstrap()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"服务未就绪：{exc}")

    return StatusResponse(
        vector_count=get_vector_count(vector_store),
        collection_name=config.collection_name,
        configured_provider=config.llm_provider,
    )


@app.get("/api/health", response_model=ApiMessage)
def health() -> ApiMessage:
    return ApiMessage(message="ok")


@app.post("/api/documents", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="请选择至少一个文件")

    config, _, vector_store = _bootstrap()
    buffered = [_UploadedFile(file.filename or "upload.bin", await file.read()) for file in files]
    result = ingest_uploads(buffered, config, vector_store)

    return UploadResponse(
        file_count=result.file_count,
        chunk_count=result.chunk_count,
        failed=list(result.failed),
    )


@app.post("/api/chat", response_model=QueryResponse)
def chat(request: QueryRequest) -> QueryResponse:
    config, _, vector_store = _bootstrap()
    runtime, runtime_api_key = _build_runtime(request, config)
    memory = _get_memory(request.session_id, runtime.memory_window)

    if get_vector_count(vector_store) == 0:
        return QueryResponse(
            session_id=request.session_id,
            answer="知识库为空，请先上传并导入企业文档。",
            sources=[],
            top_k=runtime.retriever_top_k,
            candidate_k=runtime.retriever_top_k * runtime.reranker_candidate_multiplier,
            retrieved_count=0,
            returned_count=0,
            reranker_enabled=runtime.reranker_enabled,
        )

    llm = _get_llm(runtime, runtime_api_key)
    qa_chain = build_qa_chain(
        llm=llm,
        vector_store=vector_store,
        memory=memory,
        top_k=runtime.retriever_top_k,
        reranker_candidate_multiplier=runtime.reranker_candidate_multiplier,
        reranker_top_k=runtime.reranker_top_k,
        enable_reranker=runtime.reranker_enabled,
        reranker_alpha=runtime.reranker_alpha,
    )
    result = answer_with_sources(qa_chain, request.question)

    return QueryResponse(
        session_id=request.session_id,
        answer=result.answer,
        sources=[SourceItem(**item) for item in result.sources],
        top_k=runtime.retriever_top_k,
        candidate_k=runtime.retriever_top_k * runtime.reranker_candidate_multiplier,
        retrieved_count=result.retrieved_count,
        returned_count=result.returned_count,
        reranker_enabled=runtime.reranker_enabled,
    )


@app.delete("/api/sessions/{session_id}", response_model=ApiMessage)
def clear_session(session_id: str) -> ApiMessage:
    if session_id in MEMORY_CACHE:
        MEMORY_CACHE.pop(session_id)
    return ApiMessage(message=f"会话 {session_id} 已清空")


@app.post("/api/knowledge/clear", response_model=ApiMessage)
def clear_knowledge() -> ApiMessage:
    if BASE_CONFIG is None or EMBEDDING_MODEL is None:
        _bootstrap()

    global VECTOR_STORE
    if VECTOR_STORE is None:
        raise HTTPException(status_code=503, detail="向量库未就绪")

    config = BASE_CONFIG
    VECTOR_STORE = clear_vector_store(config, EMBEDDING_MODEL)
    MEMORY_CACHE.clear()
    return ApiMessage(message="知识库已清空")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chainknowledge.api.main:app", host="0.0.0.0", port=8000, reload=False)
