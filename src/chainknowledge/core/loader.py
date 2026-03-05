from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)

from chainknowledge.core.config import DEFAULT_ALLOWED_EXTENSIONS


def supported_extensions() -> tuple[str, ...]:
    return DEFAULT_ALLOWED_EXTENSIONS


def load_documents(file_path: Path) -> List[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in {".txt", ".md", ".log", ".yml", ".yaml", ".ini", ".cfg", ".json"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext in {".docx", ".doc", ".docm"}:
        loader = Docx2txtLoader(str(path))
    elif ext == ".csv":
        loader = CSVLoader(str(path))
    elif ext in {".htm", ".html"}:
        loader = BSHTMLLoader(str(path))
    else:
        # 对其他文本类格式做兜底尝试，避免上传失败。
        loader = TextLoader(str(path), encoding="utf-8")

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
    return docs
