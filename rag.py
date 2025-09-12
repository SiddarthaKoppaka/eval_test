from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain core + memory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Vector store (persisted)
from langchain_chroma import Chroma

# Bedrock (LLM + embeddings)
from langchain_aws import ChatBedrock, BedrockEmbeddings


# ----------------------------
# Config
# ----------------------------
load_dotenv()

CHROMA_PATH = Path("chroma")                # must already exist from your index build
COLLECTION_NAME = "pdf_chunks"

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_CHAT_MODEL_ID = os.getenv(
    "BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL", "cohere.embed-english-v3")


# ----------------------------
# Vector store / retriever
# ----------------------------
def get_retriever(k: int = 4):
    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=BEDROCK_REGION)
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )
    # MMR is a good default for diverse passages; switch to "similarity" if preferred
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5})


def format_docs(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"{d.page_content}\n\n[Source: {src}, page {page}]")
    return "\n\n---\n\n".join(blocks)


def get_llm():
    return ChatBedrock(
        model_id=BEDROCK_CHAT_MODEL_ID,
        region_name=BEDROCK_REGION,
        # credentials are taken from env/role by default
    )


# ----------------------------
# Build a memory-aware, multi-turn RAG chain
# ----------------------------
def build_chat_rag_chain(k: int = 4):
    retriever = get_retriever(k=k)
    llm = get_llm()

    # 1) Condense the user's new turn into a standalone question using chat history
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's latest message into a single, standalone question using the chat history."),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    # 2) Main QA prompt that uses BOTH history and retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise assistant. Use ONLY the provided context to answer.\n"
         "If the answer isn't in the context, say you don't know.\n"
         "Cite sources as [Source: file, page N]. Be concise."),
        MessagesPlaceholder("history"),
        ("human", "Question:\n{question}\n\nContext:\n{context}")
    ])

    # Wire it up with LCEL
    base_chain = (
        {
            "question": {"history": RunnablePassthrough(), "input": RunnablePassthrough()}  # pass both into condenser
                        | condense_chain,
            "history": RunnablePassthrough(),
        }
        | {
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
            "context": lambda x: format_docs(retriever.get_relevant_documents(x["question"]))
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # 3) Add memory via RunnableWithMessageHistory.
    #
    # We'll keep a per-session buffer (ConversationBuffer) implemented by
    # InMemoryChatMessageHistory. You can swap this to Redis for production.
    store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # The wrapper will:
    # - Read "session_id" from config={"configurable": {"session_id": ...}}
    # - Append the incoming HumanMessage and resulting AIMessage to the buffer.
    chat_chain_with_memory = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",      # our incoming user text key
        history_messages_key="history",  # prompt uses MessagesPlaceholder("history")
    )

    return chat_chain_with_memory


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="Multi-Turn RAG over Chroma (LangChain + Bedrock)")

# CORS for local dev / web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    input: str
    k: int = 4


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    ok = CHROMA_PATH.exists()
    return {"status": "ok" if ok else "missing_chroma", "chroma_path": str(CHROMA_PATH.resolve())}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Build the chain (fast — light objects; if you prefer, construct once and cache globally)
    rag = build_chat_rag_chain(k=req.k)

    # IMPORTANT: pass session_id via config->configurable so memory is per-session
    answer = rag.invoke(
        {"input": req.input},
        config={"configurable": {"session_id": req.session_id}}
    )
    return ChatResponse(answer=answer)


# For local testing:  uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
