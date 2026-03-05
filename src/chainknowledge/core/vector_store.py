from __future__ import annotations

from pathlib import Path
import shutil

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from chainknowledge.core.config import AppConfig


def create_or_load_vector_store(embedding_model, config: AppConfig) -> Chroma:
    config.ensure_paths()
    return Chroma(
        collection_name=config.collection_name,
        embedding_function=embedding_model,
        persist_directory=str(config.chroma_persist_dir),
    )


def get_retriever(vector_store: Chroma, top_k: int):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def add_documents(vector_store: Chroma, documents: list[Document]) -> None:
    if documents:
        vector_store.add_documents(documents)
        vector_store.persist()


def get_vector_count(vector_store: Chroma) -> int:
    try:
        return len(vector_store.get()["ids"])
    except Exception:
        return 0


def clear_vector_store(config: AppConfig, embedding_model) -> Chroma:
    if config.chroma_persist_dir.exists():
        shutil.rmtree(config.chroma_persist_dir)
    return create_or_load_vector_store(embedding_model, config)
