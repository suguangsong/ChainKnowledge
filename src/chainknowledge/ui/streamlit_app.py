from __future__ import annotations

from dataclasses import replace
import streamlit as st
from dotenv import load_dotenv

from chainknowledge.core.config import AppConfig
from chainknowledge.core.llm import create_chat_model, create_embedding_model
from chainknowledge.core.memory import create_memory
from chainknowledge.core.vector_store import create_or_load_vector_store, clear_vector_store, get_vector_count
from chainknowledge.services.ingestion import ingest_uploads
from chainknowledge.services.qa_service import answer_with_sources, build_qa_chain
from chainknowledge.core.loader import supported_extensions


def _safe_model_text(config: AppConfig, provider: str) -> str:
    if provider == "chatglm":
        return config.chatglm_model
    return config.openai_model


def _safe_api_key(config: AppConfig, provider: str) -> str:
    if provider == "chatglm":
        return config.streamlit_chatglm_api_key
    return config.streamlit_openai_api_key


def _safe_api_base(config: AppConfig, provider: str) -> str:
    if provider == "chatglm":
        return config.chatglm_api_base
    return config.openai_api_base


def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "memory_window" not in st.session_state:
        st.session_state.memory_window = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_provider" not in st.session_state:
        st.session_state.last_provider = None


def _render_sources(sources):
    for item in sources:
        with st.expander(f"{item['idx']}. {item['file']}"):
            page = item["page"]
            if page:
                st.markdown(f"**页码/位置**：{page}")
            st.caption(item["snippet"])


def build_sidebar(config: AppConfig):
    with st.sidebar:
        st.header("企业知识库参数")
        top_k = st.slider("检索上下文段落数", min_value=1, max_value=10, value=config.retriever_top_k)
        reranker_candidate_multiplier = st.slider("重排候选放大倍率", min_value=1, max_value=8, value=config.reranker_candidate_multiplier)
        reranker_top_k = st.slider("重排后保留段落数", min_value=1, max_value=10, value=config.reranker_top_k)
        reranker_enabled = st.toggle("启用检索重排", value=config.reranker_enabled)
        reranker_alpha = st.slider(
            "检索重排权重（0-1，越高越信任向量分数）",
            min_value=0.0,
            max_value=1.0,
            value=float(config.reranker_alpha),
            step=0.05,
            format="%.2f",
        )
        chunk_size = st.slider("分块长度", min_value=300, max_value=2000, value=config.chunk_size, step=50)
        chunk_overlap = st.slider("分块重叠", min_value=0, max_value=400, value=config.chunk_overlap, step=20)
        memory_window = st.slider("记忆窗口", min_value=1, max_value=30, value=config.memory_window)

        st.divider()
        st.header("模型与密钥")
        provider = st.selectbox("LLM 提供商", options=["openai", "chatglm"], index=0 if config.llm_provider == "openai" else 1)
        model = st.text_input("模型名", value=_safe_model_text(config, provider), placeholder="输入 ChatGLM/OpenAI 模型名")
        api_base = st.text_input("API Base", value=_safe_api_base(config, provider), placeholder="可选，兼容 OpenAI 协议的 Base URL")
        api_key = st.text_input("API Key", value="", type="password", placeholder="留空则使用 .env 配置")

        if api_key:
            st.caption("本次运行已使用页面手输密钥")

        st.divider()
        if st.button("清空知识库"):
            st.session_state.clear_knowledge_requested = True

        if st.button("清空对话"):
            st.session_state.messages = []
            st.session_state.memory = create_memory(memory_window)
            st.rerun()

        st.caption(f"当前模型配置：{provider} / {model}")

    runtime_api_key = (api_key or _safe_api_key(config, provider)).strip()

    runtime = replace(
        config,
        llm_provider=provider,
        openai_model=model,
        chatglm_model=model,
        openai_api_base=api_base,
        chatglm_api_base=api_base,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retriever_top_k=top_k,
        reranker_candidate_multiplier=reranker_candidate_multiplier,
        reranker_top_k=reranker_top_k,
        reranker_enabled=reranker_enabled,
        reranker_alpha=reranker_alpha,
        memory_window=memory_window,
    )
    return runtime, runtime_api_key


def main() -> None:
    load_dotenv()
    config = AppConfig.from_env()
    config.ensure_paths()

    st.set_page_config(page_title="ChainKnowledge", page_icon="📚", layout="wide")
    st.title("ChainKnowledge 企业知识库问答")
    st.caption("基于 LangChain + Chroma + OpenAI/ChatGLM + Streamlit 构建的内部知识问答示例")

    init_session()
    runtime, runtime_api_key = build_sidebar(config)

    runtime_signature = (
        runtime.llm_provider,
        runtime.openai_model,
        runtime.chatglm_model,
        runtime.openai_api_base,
        runtime.chatglm_api_base,
        runtime.chunk_size,
        runtime.chunk_overlap,
        runtime.retriever_top_k,
        runtime.reranker_candidate_multiplier,
        runtime.reranker_top_k,
        runtime.reranker_alpha,
        runtime.reranker_enabled,
        runtime.memory_window,
        runtime_api_key,
    )
    if st.session_state.get("runtime_signature") != runtime_signature:
        st.session_state.qa_chain = None
        st.session_state.runtime_signature = runtime_signature

    if st.session_state.memory is None or st.session_state.memory_window != runtime.memory_window:
        st.session_state.memory = create_memory(runtime.memory_window)
        st.session_state.memory_window = runtime.memory_window

    try:
        embedding_model = create_embedding_model(
            config,
            api_key_override=config.streamlit_openai_api_key,
            base_url_override=config.openai_api_base,
        )
        vector_store = create_or_load_vector_store(embedding_model, config)
    except Exception as exc:
        st.error(f"向量模型初始化失败：{exc}")
        return

    if st.session_state.get("clear_knowledge_requested", False):
        del st.session_state.clear_knowledge_requested
        try:
            vector_store = clear_vector_store(config, embedding_model)
            st.success("知识库已清空")
            st.rerun()
        except Exception as exc:
            st.error(f"清空知识库失败：{exc}")
            return

    st.session_state.vector_store = vector_store
    index_count = get_vector_count(vector_store)

    st.subheader("1. 上传企业文档")
    uploaded_files = st.file_uploader(
        "上传文件（支持 PDF、DOCX、TXT、MD、CSV、HTML）",
        type=[ext.lstrip(".") for ext in supported_extensions()],
        accept_multiple_files=True,
    )

    if st.button("导入到知识库"):
        if not uploaded_files:
            st.warning("请先选择要上传的文件")
        else:
            with st.spinner("文档入库中，正在执行：加载 -> 分片 -> 向量化..."):
                result = ingest_uploads(uploaded_files, runtime, vector_store)
            if result.file_count:
                st.success(f"已导入 {result.file_count} 个文档，生成 {result.chunk_count} 个分片")
            if result.failed:
                for msg in result.failed:
                    st.warning(msg)
            index_count = get_vector_count(vector_store)
            st.info(f"当前知识库共 {index_count} 个向量片段")

    st.subheader("2. 多轮知识库问答")
    st.caption(
        f"检索参数：TopK={runtime.retriever_top_k}，候选检索数={runtime.retriever_top_k * runtime.reranker_candidate_multiplier}"
        f"，重排后TopK={runtime.reranker_top_k}，重排{'开启' if runtime.reranker_enabled else '关闭'}"
    )
    if index_count == 0:
        st.info("知识库当前为空，请先上传并导入文档再进行问答")

    if st.session_state.qa_chain is None:
        try:
            llm = create_chat_model(
                runtime,
                provider=runtime.llm_provider,
                api_key_override=runtime_api_key,
                model_override=runtime.openai_model if runtime.llm_provider == "openai" else runtime.chatglm_model,
                base_url_override=runtime.openai_api_base if runtime.llm_provider == "openai" else runtime.chatglm_api_base,
            )
            st.session_state.qa_chain = build_qa_chain(
                llm=llm,
                vector_store=vector_store,
                memory=st.session_state.memory,
                top_k=runtime.retriever_top_k,
                reranker_candidate_multiplier=runtime.reranker_candidate_multiplier,
                reranker_top_k=runtime.reranker_top_k,
                enable_reranker=runtime.reranker_enabled,
                reranker_alpha=runtime.reranker_alpha,
            )
        except Exception as exc:
            st.error(f"模型初始化失败：{exc}")
            return

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                _render_sources(item["sources"])

    question = st.chat_input("你想从知识库里查什么？（例如：公司请假流程、接口上线标准、SOP 要点）")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if get_vector_count(vector_store) == 0:
                st.warning("知识库为空，先导入文档再提问")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "知识库中还没有内容，请先上传企业文档。",
                        "sources": [],
                    }
                )
            else:
                with st.spinner("检索中，生成基于证据的回答..."):
                    result = answer_with_sources(st.session_state.qa_chain, question)
                answer = result.answer.strip()
                if not answer:
                    answer = "当前文档中未检索到可直接支持该问题的内容，我建议换一个更具体的问法。"
                st.markdown(answer)
                if result.sources:
                    st.caption("回答溯源")
                    _render_sources(result.sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": result.sources})

        st.session_state.messages = st.session_state.messages[-config.max_recent_pairs :]


if __name__ == "__main__":
    main()
