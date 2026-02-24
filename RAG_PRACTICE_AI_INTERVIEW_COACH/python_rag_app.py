#!/usr/bin/env python3
"""Simple CLI RAG app for your AI Resume & Interview Coach project.

Usage (from this folder):

    cd ~/.openclaw/workspace/repos/LLM_Project/RAG_PRACTICE_AI_INTERVIEW_COACH
    python3 python_rag_app.py

It will:
- Load OPENAI_API_KEY from ../llm_engineering/.env (same as the notebook)
- Build / reuse a Chroma vector DB under ./vector_db
- Start a simple question-answer loop in the terminal.
"""

import os
import glob

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory

MODEL = "gpt-4o-mini"
DB_NAME = "vector_db"
KNOWLEDGE_BASE_DIR = "knowledge-base"


def load_env() -> None:
    """Load environment variables from ../llm_engineering/.env.

    This matches the logic in the notebook.
    """

    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dotenv_path = os.path.join(project_root, "llm_engineering", ".env")

    if not os.path.exists(dotenv_path):
        print(f"[WARN] .env file not found at: {dotenv_path}")
    else:
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print(f"[INFO] Loaded environment variables from {dotenv_path}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please add it to llm_engineering/.env or your shell env."
        )


def load_documents() -> list:
    """Load markdown documents from the knowledge-base folder.

    Adds a doc_type metadata based on the subfolder name (Essay / RESUME / Technical_Skills).
    """

    folders = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*"))
    if not folders:
        raise RuntimeError(
            f"No subfolders found under {KNOWLEDGE_BASE_DIR}/. "
            "Expected e.g. Essay, RESUME, Technical_Skills."
        )

    def add_metadata(doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc

    text_loader_kwargs = {"encoding": "utf-8"}

    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs,
        )
        folder_docs = loader.load()
        documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])

    if not documents:
        raise RuntimeError(
            f"No markdown files found under {KNOWLEDGE_BASE_DIR}/ subfolders."
        )

    print(f"[INFO] Loaded {len(documents)} documents from {KNOWLEDGE_BASE_DIR}/")
    return documents


def build_vectorstore(documents):
    """Split documents into chunks and build a Chroma vector store.

    If a previous DB exists, it will be replaced (same as the notebook).
    """

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings()

    if os.path.exists(DB_NAME):
        print(f"[INFO] Existing vector DB '{DB_NAME}' found; deleting collection and rebuilding.")
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )
    print(f"[INFO] Vectorstore created with {vectorstore._collection.count()} documents.")

    return vectorstore


def build_rag_chain(vectorstore):
    """Create a modern LangChain RAG pipeline with chat history.

    Uses retriever | prompt | llm, wrapped in RunnableWithMessageHistory.
    """

    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an AI resume and interview coach for Beau. "
        "Use the retrieved chunks from Beau's resume, essays, and technical skills to answer "
        "questions about his background, generate tailored resume bullet points, and suggest "
        "interview talking points. Be specific and refer to concrete experiences when possible."
    )

    # Include retrieved context explicitly in the system prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "\n\nUse the following context from Beau's documents when helpful:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model=MODEL, temperature=0.2)

    # Simple in‑memory chat history storage keyed by session id
    store: dict[str, ChatMessageHistory] = {}

    def get_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # Base RAG pipeline: retriever -> format context -> prompt -> llm
    rag_chain = (
        RunnableLambda(
            lambda x: {
                "question": x["question"],
                "chat_history": x.get("chat_history", []),
                "context_docs": retriever.invoke(x["question"]),
            }
        )
        | RunnableLambda(
            lambda x: {
                "question": x["question"],
                "chat_history": x["chat_history"],
                "context": "\n\n".join(d.page_content for d in x["context_docs"]),
            }
        )
        | prompt
        | llm
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain_with_history


def main() -> None:
    print("=== AI Resume RAG & Interview Coach (CLI) ===")
    print("This will build/use a local vector DB under './vector_db'.")
    print("Press Ctrl+C or type 'exit' to quit.\n")

    load_env()
    documents = load_documents()
    vectorstore = build_vectorstore(documents)
    chain = build_rag_chain(vectorstore)

    while True:
        try:
            question = input("\nAsk a question about your resume / interviews (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Goodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("[INFO] Goodbye!")
            break

        try:
            result = chain.invoke({"question": question}, config={"configurable": {"session_id": "cli"}})
        except Exception as e:
            print(f"[ERROR] Failed to run RAG chain: {e}")
            continue

        answer = str(result)
        print("\n=== Answer ===")
        print(answer)


if __name__ == "__main__":
    main()
