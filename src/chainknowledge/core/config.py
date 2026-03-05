from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWED_EXTENSIONS = (
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".doc",
    ".docm",
    ".html",
    ".htm",
    ".csv",
)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    openai_api_key: str
    api_openai_api_key: str
    openai_api_base: str
    openai_model: str
    openai_embedding_model: str
    chatglm_api_key: str
    api_chatglm_api_key: str
    chatglm_api_base: str
    chatglm_model: str
    streamlit_openai_api_key: str
    streamlit_chatglm_api_key: str
    chunk_size: int
    chunk_overlap: int
    retriever_top_k: int
    reranker_candidate_multiplier: int
    reranker_top_k: int
    reranker_alpha: float
    reranker_enabled: bool
    memory_window: int
    chroma_persist_dir: Path
    upload_dir: Path
    collection_name: str
    max_upload_size: int
    allowed_extensions: Tuple[str, ...]
    max_recent_pairs: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        api_openai_api_key = _env("API_OPENAI_API_KEY") or _env("OPENAI_API_KEY")
        api_chatglm_api_key = _env("API_CHATGLM_API_KEY") or _env("CHATGLM_API_KEY")
        streamlit_openai_api_key = _env("STREAMLIT_OPENAI_API_KEY") or api_openai_api_key
        streamlit_chatglm_api_key = _env("STREAMLIT_CHATGLM_API_KEY") or api_chatglm_api_key

        return cls(
            llm_provider=_env("LLM_PROVIDER", "openai").lower(),
            openai_api_key=api_openai_api_key,
            api_openai_api_key=api_openai_api_key,
            openai_api_base=_env("OPENAI_API_BASE"),
            openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
            openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            chatglm_api_key=api_chatglm_api_key,
            api_chatglm_api_key=api_chatglm_api_key,
            chatglm_api_base=_env("CHATGLM_API_BASE"),
            chatglm_model=_env("CHATGLM_MODEL", "glm-4"),
            streamlit_openai_api_key=streamlit_openai_api_key,
            streamlit_chatglm_api_key=streamlit_chatglm_api_key,
            chunk_size=_env_int("CHUNK_SIZE", 1000),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 180),
            retriever_top_k=_env_int("RETRIEVER_TOP_K", 4),
            reranker_candidate_multiplier=_env_int("RERANK_CANDIDATE_MULTIPLIER", 3),
            reranker_top_k=_env_int("RERANK_TOP_K", 4),
            reranker_alpha=_env_float("RERANK_ALPHA", 0.7),
            reranker_enabled=_env_bool("RERANK_ENABLED", True),
            memory_window=_env_int("MEMORY_WINDOW", 6),
            chroma_persist_dir=_resolve_path(_env("CHROMA_PERSIST_DIR", "storage/chroma")),
            upload_dir=_resolve_path(_env("UPLOAD_DIR", "storage/uploads")),
            collection_name=_env("COLLECTION_NAME", "chainknowledge"),
            max_upload_size=_env_int("MAX_UPLOAD_SIZE_MB", 40) * 1024 * 1024,
            allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS,
            max_recent_pairs=_env_int("MAX_RECENT_PAIRS", 30),
        )

    def ensure_paths(self) -> None:
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
