from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------------- Config ----------------
load_dotenv()

CHROMA_PATH = Path("chroma")
COLLECTION_NAME = "pdf_chunks"

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL", "cohere.embed-english-v3")
BEDROCK_CHAT_MODEL_ID = os.getenv(
    "BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)

# ---------------- Vectorstore ----------------
def get_retriever(k: int = 4):
    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=BEDROCK_REGION)
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

# ---------------- LLM ----------------
def get_llm():
    return ChatBedrock(model_id=BEDROCK_CHAT_MODEL_ID, region_name=BEDROCK_REGION)

# ---------------- Memory + Chain ----------------
memory_store = {}  # per-session

def get_chain(session_id: str, k: int = 4):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    retriever = get_retriever(k)
    llm = get_llm()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory_store[session_id],
        return_source_documents=True,
    )
    return chain

# ---------------- FastAPI ----------------
app = FastAPI(title="Simple RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chain = get_chain(req.session_id, req.k)
    result = chain.invoke({"question": req.input})
    return ChatResponse(answer=result["answer"])

@app.get("/health")
def health():
    return {"status": "ok", "chroma_exists": CHROMA_PATH.exists()}


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ---------------- Config ----------------
load_dotenv()

CHROMA_PATH = Path("chroma")
COLLECTION_NAME = "pdf_chunks"

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL", "cohere.embed-english-v3")
BEDROCK_CHAT_MODEL_ID = os.getenv(
    "BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)

# ---------------- Vectorstore ----------------
def get_retriever(k: int = 4):
    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=BEDROCK_REGION)
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

# ---------------- LLM ----------------
def get_llm():
    return ChatBedrock(model_id=BEDROCK_CHAT_MODEL_ID, region_name=BEDROCK_REGION)

# ---------------- Memory + Chain ----------------
memory_store = {}  # per-session

def get_chain(session_id: str, k: int = 4):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    retriever = get_retriever(k)
    llm = get_llm()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory_store[session_id],
        return_source_documents=True,
    )
    return chain

# ---------------- FastAPI ----------------
app = FastAPI(title="Simple RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chain = get_chain(req.session_id, req.k)
    result = chain.invoke({"question": req.input})
    return ChatResponse(answer=result["answer"])

@app.get("/health")
def health():
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Prompt used to generate the final answer from retrieved docs
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
     "If the answer is not in the context, say you don't know.\n\n"
     "Constraints:\n"
     "- Be concise.\n"
     "- Cite sources by number like [1], [2] using the provided documents' order.\n"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", 
     "Context:\n{context}\n\n"
     "Question: {question}\n\n"
     "Answer:")
])

# Prompt used to rewrite a follow-up into a standalone question (for better retrieval)
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "Rewrite the user's latest question into a standalone question, using the chat history for context. "
     "Do not answer the question; only rewrite it."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Latest user question: {question}")
])
