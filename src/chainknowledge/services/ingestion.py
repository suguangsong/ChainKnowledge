from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from chainknowledge.core.config import AppConfig
from chainknowledge.core.loader import load_documents
from chainknowledge.core.splitter import split_documents


@dataclass(frozen=True)
class IngestResult:
    chunk_count: int
    file_count: int
    failed: tuple[str, ...]


def _save_uploaded_file(uploaded_file, upload_dir: Path) -> Tuple[Path, str]:
    safe_name = f"{uuid.uuid4().hex}_{Path(uploaded_file.name).name}"
    target = upload_dir / safe_name

    with open(target, "wb") as f:
        f.write(uploaded_file.getvalue())

    return target, uploaded_file.name


def _attach_metadata(documents: List[Document], source_name: str, source_path: Path) -> List[Document]:
    for doc in documents:
        doc.metadata.setdefault("file_name", source_name)
        doc.metadata.setdefault("source", str(source_path))
        doc.metadata.setdefault("upload_name", source_name)
        doc.metadata.setdefault("source_abs_path", str(source_path))
    return documents


def ingest_uploads(
    uploaded_files: Iterable,
    config: AppConfig,
    vector_store: Chroma,
) -> IngestResult:
    allowed = set(config.allowed_extensions)
    total_files = 0
    total_chunks = 0
    failed: List[str] = []

    for uploaded in uploaded_files:
        original_name = Path(uploaded.name).name
        ext = Path(original_name).suffix.lower()

        if ext not in allowed:
            failed.append(f"{original_name}（暂不支持该格式）")
            continue

        if hasattr(uploaded, "size") and uploaded.size > config.max_upload_size:
            failed.append(f"{original_name}（超过文件体积上限）")
            continue

        saved_path, original_name = _save_uploaded_file(uploaded, config.upload_dir)
        try:
            docs = load_documents(saved_path)
            docs = _attach_metadata(docs, original_name, saved_path)
            chunked = split_documents(docs, config.chunk_size, config.chunk_overlap)
        except Exception as exc:
            failed.append(f"{original_name}（解析失败：{exc}）")
            continue

        vector_store.add_documents(chunked)
        total_files += 1
        total_chunks += len(chunked)

    if total_files:
        vector_store.persist()

    return IngestResult(
        chunk_count=total_chunks,
        file_count=total_files,
        failed=tuple(failed),
    )
