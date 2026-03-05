from __future__ import annotations

from langchain.memory import ConversationBufferWindowMemory


def create_memory(window: int):
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=window,
    )
