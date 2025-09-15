import os
import uuid
import time
import math
import httpx
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from nltk import download as nltk_download
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,            # groundedness (higher = more grounded)
    answer_relevancy,        # how relevant answer is to question
    context_precision,       # fraction of retrieved that are needed
    context_recall,          # how much of needed info was retrieved
    answer_correctness,      # LLM-based correctness vs ground truth
)

from langchain_aws import ChatBedrock, BedrockEmbeddings

# ---------------- Config ----------------
load_dotenv()

# Bedrock config (falls back to sensible defaults)
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_CHAT_MODEL_ID = os.getenv(
    "BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
)
BEDROCK_EMBED_MODEL_ID = os.getenv(
    "BEDROCK_EMBED_MODEL", "cohere.embed-english-v3"
)

# Build Bedrock LLM + embeddings once
def get_llm():
    return ChatBedrock(model_id=BEDROCK_CHAT_MODEL_ID, region_name=BEDROCK_REGION)

def get_embeddings():
    return BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL_ID, region_name=BEDROCK_REGION)

LLM = get_llm()
EMB = get_embeddings()

# NLTK bootstrap
try:
    nltk_download("punkt", quiet=True)
    nltk_download("punkt_tab", quiet=True)  # newer nltk splits this
except Exception:
    pass

# ---------------- FastAPI ----------------
app = FastAPI(title="RAG Eval Service (RAGAS + Bedrock)")

class EvalRequest(BaseModel):
    rag_api_url: str                 # e.g., http://localhost:8000/chat
    dataset_csv_path: str            # CSV with columns: question, ground_truth
    session_prefix: str = "eval"
    k: int = 4
    return_sources: bool = True      # ask RAG app for sources/contexts if supported
    headers: Dict[str, str] | None = None   # auth etc.
    max_rows: Optional[int] = None   # for quick smoke tests

class EvalResponse(BaseModel):
    summary: Dict[str, float]
    bleu_avg: float
    compliance_avg: float
    details: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok", "bedrock_region": BEDROCK_REGION}

def call_rag(
    url: str,
    question: str,
    session_id: str,
    k: int,
    return_sources: bool,
    headers: Optional[Dict[str, str]] = None
) -> Tuple[str, List[str]]:
    """
    Call the RAG /chat endpoint. Expect JSON with 'answer' and optional 'sources'.
    Fallback: if no sources, return empty list.
    """
    payload = {
        "session_id": session_id,
        "input": question,
        "k": k,
        "return_sources": return_sources,
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload, headers=headers or {})
        r.raise_for_status()
        data = r.json()
    answer = data.get("answer", "")
    sources = data.get("sources") or []
    # coerce sources to strings
    sources = [str(s) for s in sources]
    return answer, sources

def compute_bleu(ref: str, hyp: str) -> float:
    """
    Simple BLEU-4 with smoothing. Tokenize with NLTK.
    """
    ref_tokens = word_tokenize(ref.lower()) if ref else []
    hyp_tokens = word_tokenize(hyp.lower()) if hyp else []
    if not ref_tokens or not hyp_tokens:
        return 0.0
    smoothie = SmoothingFunction().method3
    try:
        score = float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie))
    except ZeroDivisionError:
        score = 0.0
    # keep in [0,1]
    return max(0.0, min(score, 1.0))

PROMPT_COMPLIANCE_RUBRIC = """You are grading whether the assistant's answer complies with the system/user prompt style and structure rules.
Rate on a 1-5 scale (integers only), where:
1 = non-compliant, 3 = partially compliant, 5 = fully compliant.

Consider:
- Did the answer follow the requested format/structure?
- Did it avoid prohibited content per the prompt/corpus policy?
- Did it respect persona/tone/length/sections (if specified)?
- Did it include or omit elements as directed (e.g., no sources if asked; required headings; JSON shape etc.)?

Return a strict JSON object with keys:
"score": int (1-5),
"rationale": string (one or two sentences).
"""

def judge_compliance(question: str, answer: str) -> Dict[str, Any]:
    """
    Use Bedrock LLM as judge; returns {"score": int, "rationale": str}
    """
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Evaluate compliance per the rubric."
    )
    resp = LLM.invoke([
        ("system", PROMPT_COMPLIANCE_RUBRIC),
        ("user", user_msg),
    ])
    text = resp.content if hasattr(resp, "content") else str(resp)
    # naive JSON guard
    import json
    try:
        out = json.loads(text)
        score = int(out.get("score", 3))
        rationale = str(out.get("rationale", "")).strip()
    except Exception:
        score, rationale = 3, "Could not parse judge output."
    score = max(1, min(5, score))
    return {"score": score, "rationale": rationale}

@app.post("/eval", response_model=EvalResponse)
def run_eval(req: EvalRequest):
    # 1) Load dataset
    df = pd.read_csv(req.dataset_csv_path)
    # Expect columns: question, ground_truth
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("CSV must include columns: 'question', 'ground_truth'")
    if req.max_rows:
        df = df.head(req.max_rows)

    # 2) Query RAG for answers (+contexts)
    rows: List[Dict[str, Any]] = []
    session_base = f"{req.session_prefix}-{int(time.time())}"
    for i, row in df.iterrows():
        q = str(row["question"])
        gt = str(row["ground_truth"])
        session_id = f"{session_base}-{i}-{uuid.uuid4().hex[:6]}"
        try:
            ans, ctxs = call_rag(
                url=req.rag_api_url,
                question=q,
                session_id=session_id,
                k=req.k,
                return_sources=req.return_sources,
                headers=req.headers,
            )
        except Exception as e:
            ans, ctxs = "", []
        rows.append({
            "question": q,
            "ground_truth": gt,
            "answer": ans,
            "contexts": ctxs,
        })

    # 3) Build HuggingFace Dataset for RAGAS
    ds = Dataset.from_list(rows)

    # 4) RAGAS evaluation (Bedrock LLM+Embeddings)
    #    You can pick any subset; here we compute a balanced set of metrics:
    ragas_result = evaluate(
        ds,
        metrics=[
            faithfulness,        # aka groundedness (↑ better)
            answer_relevancy,    # (↑ better)
            context_precision,   # (↑ better)
            context_recall,      # (↑ better)
            answer_correctness,  # (↑ better) Truthfulness vs GT
        ],
        llm=LLM,
        embeddings=EMB,
    )
    ragas_df = ragas_result.to_pandas()

    # 5) BLEU per example (vs. ground_truth)
    bleu_scores = []
    for r in rows:
        bleu_scores.append(compute_bleu(r["ground_truth"], r["answer"]))
    bleu_avg = float(sum(bleu_scores) / len(bleu_scores)) if bleu_scores else 0.0

    # 6) LLM-as-judge for prompt compliance
    comp_scores = []
    comp_rationales = []
    for r in rows:
        judge = judge_compliance(r["question"], r["answer"])
        comp_scores.append(judge["score"])
        comp_rationales.append(judge["rationale"])
    compliance_avg = float(sum(comp_scores) / len(comp_scores)) if comp_scores else 0.0

    # 7) Assemble per-sample details
    details = []
    for i, r in enumerate(rows):
        drow = {
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "answer": r["answer"],
            "contexts_count": len(r["contexts"]),
            "bleu": bleu_scores[i],
            "compliance_score": comp_scores[i],
            "compliance_note": comp_rationales[i],
        }
        # merge ragas columns for the same row index
        for col in ragas_df.columns:
            # ragas columns are per-row metrics
            drow[col] = float(ragas_df.iloc[i][col])
        details.append(drow)

    # 8) Summary (macro averages from RAGAS + BLEU + compliance)
    summary = {}
    for col in ragas_df.columns:
        summary[col] = float(ragas_df[col].mean())
    return {
        "summary": summary,
        "bleu_avg": bleu_avg,
        "compliance_avg": compliance_avg,
        "details": details,
    }




# --- add to your RAG app ---

class ChatRequest(BaseModel):
    session_id: str
    input: str
    k: int = 4
    return_sources: bool = False  # NEW

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] | None = None  # NEW

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chain = get_chain(req.session_id, req.k)
    result = chain.invoke({"question": req.input})
    payload = {"answer": result["answer"]}
    if req.return_sources:
        # convert Documents -> strings
        payload["sources"] = [d.page_content for d in result.get("source_documents", [])]
    return payload
