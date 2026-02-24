#!/usr/bin/env python3
"""Gradio web UI for the AI Resume RAG & Interview Coach.

Usage (in your VM):

    cd ~/.openclaw/workspace
    source rag-venv/bin/activate
    cd repos/LLM_Project/RAG_PRACTICE_AI_INTERVIEW_COACH
    python3 rag_gradio_app.py

Then open the URL printed by Gradio (usually http://127.0.0.1:7860) in your browser.
"""

import os
from typing import List, Tuple

import gradio as gr

from python_rag_app import (
    load_env,
    load_documents,
    build_vectorstore,
    build_rag_chain,
)


def build_chain():
    """Initialize env, vectorstore, and RAG chain once at startup."""
    load_env()
    documents = load_documents()
    vectorstore = build_vectorstore(documents)
    chain = build_rag_chain(vectorstore)
    return chain


# Build once when the script starts
CHAIN = build_chain()


def chat_fn(message: str, history: List[Tuple[str, str]]) -> str:
    """Gradio chat handler.

    - `message` is the latest user input
    - `history` is a list of (user, bot) tuples, but our RAG chain already
      manages its own chat history via RunnableWithMessageHistory.
    """
    try:
        result = CHAIN.invoke(
            {"question": message},
            config={"configurable": {"session_id": "gradio"}},
        )
    except Exception as e:
        return f"[ERROR] Failed to run RAG chain: {e}"

    # result is usually a ChatMessage; prefer .content if present
    answer = getattr(result, "content", None)
    if answer is None:
        answer = str(result)
    return answer


def main() -> None:
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="AI Resume RAG & Interview Coach (RAG)",
        description="Ask questions about your resume, technical skills, and interview preparation.",
    )

    # share=False keeps it local to your VM (127.0.0.1)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
