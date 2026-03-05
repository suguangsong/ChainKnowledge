from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chainknowledge.core.reranker import create_reranker

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是 ChainKnowledge 企业知识库问答助手，只能基于检索资料回答问题。\n\n"
            "规则：\n"
            "1. 回答必须有依据，不能凭空编造。\n"
            "2. 有明确业务限制时需在回复中保留边界。\n"
            "3. 只基于检索文本，不要引用你不知道的内容。\n"
            "4. 如果信息不足，建议用户补充文档或改写问题。\n",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """以下是检索到的资料片段（按相关性排序）：\n\n{context}\n\n"
            "用户问题：{question}\n\n"
            "请给出可直接用于企业内训和协作的清晰答案。""",
        ),
    ]
)


@dataclass(frozen=True)
class QAResult:
    answer: str
    sources: List[Dict[str, Any]]
    retrieved_count: int
    returned_count: int


def build_qa_chain(
    llm,
    vector_store: Chroma,
    memory,
    top_k: int,
    reranker_candidate_multiplier: int = 3,
    reranker_top_k: int | None = None,
    enable_reranker: bool = True,
    reranker_alpha: float = 0.7,
):
    base_top_k = max(1, int(top_k))
    multiplier = max(1, int(reranker_candidate_multiplier))
    candidate_k = base_top_k * multiplier
    final_top_k = max(1, int(reranker_top_k or base_top_k))
    final_top_k = min(final_top_k, candidate_k)

    reranker = create_reranker(enabled=enable_reranker, alpha=reranker_alpha)
    return {
        "llm": llm,
        "vector_store": vector_store,
        "memory": memory,
        "top_k": base_top_k,
        "candidate_k": candidate_k,
        "final_top_k": final_top_k,
        "reranker": reranker,
    }


def _history_from_memory(memory) -> list[BaseMessage]:
    if memory is None:
        return []

    try:
        history = memory.load_memory_variables({}).get("chat_history", [])
    except Exception:
        return []

    return history if isinstance(history, list) else []


def _build_context(documents: Sequence[Any], max_len: int = 700) -> str:
    blocks = []
    for idx, doc in enumerate(documents, start=1):
        snippet = (doc.page_content or "").replace("\n", " ").strip()
        blocks.append(f"[{idx}] {snippet[:max_len]}")
    return "\n\n".join(blocks)


def _build_sources(documents: Sequence[Any]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents, start=1):
        file_name = doc.metadata.get("file_name") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page") or doc.metadata.get("page_label")
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        sources.append(
            {
                "idx": idx,
                "file": file_name,
                "page": page,
                "snippet": snippet[:500],
            }
        )
    return sources


def answer_with_sources(qa_chain, question: str) -> QAResult:
    chain_llm = qa_chain["llm"]
    vector_store = qa_chain["vector_store"]
    memory = qa_chain["memory"]
    top_k = int(qa_chain["top_k"])
    candidate_k = int(qa_chain["candidate_k"])
    final_top_k = int(qa_chain["final_top_k"])
    reranker = qa_chain["reranker"]

    if not question.strip():
        return QAResult(answer="请先输入问题。", sources=[], retrieved_count=0, returned_count=0)

    raw_docs = vector_store.similarity_search_with_score(question, k=candidate_k)
    retrieved_count = len(raw_docs)
    if reranker is not None:
        documents = reranker.rerank(question, raw_docs, top_k=final_top_k)
    else:
        documents = [doc for doc, _ in raw_docs[:final_top_k]]
    returned_count = len(documents)

    if not documents:
        return QAResult(
            answer="当前知识库未检索到直接支持该问题的内容，请上传更相关的文档或缩小问题范围。",
            sources=[],
            retrieved_count=retrieved_count,
            returned_count=0,
        )

    context = _build_context(documents)
    chain = QA_PROMPT | chain_llm | StrOutputParser()
    answer = chain.invoke(
        {
            "chat_history": _history_from_memory(memory),
            "context": context,
            "question": question,
        }
    )
    answer_text = (answer or "").strip()

    if not answer_text:
        answer_text = "暂未生成有效答案，请稍后重试。"

    if memory is not None:
        memory.save_context({"question": question}, {"answer": answer_text})

    return QAResult(
        answer=answer_text,
        sources=_build_sources(documents),
        retrieved_count=retrieved_count,
        returned_count=returned_count,
    )
