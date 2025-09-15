# prep_insurance_qna.py
import csv
import json
import os
import uuid
from typing import List, Dict, Any
import requests

# ---- Config ----
API_URL = os.getenv("RAG_CHAT_URL", "http://localhost:5000/chat")  # /chat endpoint
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
OUT_CSV = os.getenv("OUT_CSV", "insurance_qna.csv")

# If you want one conversation (memory carries across questions), set to "1"
USE_SINGLE_SESSION = os.getenv("USE_SINGLE_SESSION", "0") == "1"

# ---- Questions (mostly insurance policy topics) ----
QUESTIONS = [
  # ——— Auto (Allstate AU127-1) ———
  "When does the Allstate auto policy apply by location and policy period?",  # :contentReference[oaicite:0]{index=0}
  "What happens if you fail to report a newly acquired or replacement auto within 60 days?",  # :contentReference[oaicite:1]{index=1}
  "Explain the rule prohibiting combining limits of two or more autos on one claim.",  # :contentReference[oaicite:2]{index=2}
  "Under Liability Coverage, who qualifies as an 'insured person' while using your insured auto or a non-owned auto?",  # :contentReference[oaicite:3]{index=3}
  "List situations excluded under Liability (e.g., carrying persons or property for a charge, certain business use).",  # :contentReference[oaicite:4]{index=4}
  "State the Liability 'each person' and 'each occurrence' limit rules and how multiple policies/vehicles affect them.",  # :contentReference[oaicite:5]{index=5}
  "When is Allstate’s liability coverage excess over other collectible insurance?",  # :contentReference[oaicite:6]{index=6}
  "What cooperation is required from an insured during investigation, settlement, or defense?",  # :contentReference[oaicite:7]{index=7}
  "Under Medical Payments, what expenses are covered and what timing rules apply?",  # :contentReference[oaicite:8]{index=8}
  "Describe Allstate’s right to contest 'unreasonable or unnecessary' medical expenses and the insured’s protections.",  # :contentReference[oaicite:9]{index=9}
  "Define who is an insured and what vehicles qualify under Uninsured Motorists (UM) coverage.",  # :contentReference[oaicite:10]{index=10}
  "Explain the UM 'each person'/'each accident' limits and how other payments reduce UM damages payable.",  # :contentReference[oaicite:11]{index=11}
  "When does UM coverage become excess, and how does it coordinate with other policies?",  # :contentReference[oaicite:12]{index=12}
  "What proof-of-claim, medical exam, and notice requirements apply under UM?",  # :contentReference[oaicite:13]{index=13}
  "What steps must an insured take after an auto accident or if sued, and what actions could prejudice coverage?",  # :contentReference[oaicite:14]{index=14}

  # ——— Renters (Shelter HO-4) ———
  "What does Coverage C (Personal Property) insure and which named perils trigger coverage?",  # :contentReference[oaicite:15]{index=15}
  "How does the policy treat water/steam discharge from plumbing/HVAC and what key exclusions apply (e.g., vacancy >30 days, seepage >14 days)?",  # :contentReference[oaicite:16]{index=16}
  "What are the freezing protections required to maintain coverage for plumbing/HVAC during vacancy or unoccupied periods?",  # :contentReference[oaicite:17]{index=17}
  "What is the 10% off-premises limit for personal property kept away from the residence for >30 consecutive days?",  # :contentReference[oaicite:18]{index=18}
  "List the special limits for categories like money, jewelry theft, silverware theft, guns, and business property.",  # :contentReference[oaicite:19]{index=19}
  "What additional coverages exist (e.g., refrigerated food, building additions/alterations) and their limits?",  # :contentReference[oaicite:20]{index=20}
  "What must the insured do immediately after a covered loss (police notice for theft, protect property, sworn proof of loss in 60 days)?",  # :contentReference[oaicite:21]{index=21}
  "Explain valuation/settlement: when will the insurer pay market value, restoration cost, or replace in kind; how does the deductible apply?",  # :contentReference[oaicite:22]{index=22}
  "Under Section II – Personal Liability, what is the 'each occurrence' limit and the special $100,000 cap for certain non-owned watercraft uses?",  # :contentReference[oaicite:23]{index=23}
  "Which motorized vehicle/aircraft/watercraft liability scenarios are excluded, and what notable exceptions exist (e.g., golf carts, small outboards)?",  # :contentReference[oaicite:24]{index=24}
  "When does the HO-4 act as excess insurance versus pro-rata with other policies?",  # :contentReference[oaicite:25]{index=25}
  "What changes must the insured report during the policy period (residence, occupants, animals, business on premises)?",  # :contentReference[oaicite:26]{index=26}
  "What are the consequences of concealment/fraud or failure to meet duties under the policy?",  # :contentReference[oaicite:27]{index=27}
  "How does the policy define 'accident', 'accidental direct physical loss', and 'personal property' for coverage purposes?",  # :contentReference[oaicite:28]{index=28}
  "What is covered under 'Damage to Property of Others' and 'Medical Payments to Others' (high-level)?",  # :contentReference[oaicite:29]{index=29}
]


def call_chat_api(question: str, session_id: str, k: int = TOP_K) -> Dict[str, Any]:
    """Call /chat with the required payload and return JSON."""
    payload = {"session_id": session_id, "input": question, "k": k}
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()  # expects: {"answer": "..."}
    except Exception as e:
        return {"error": str(e)}

def main():
    rows = []

    # Choose session strategy
    global_session = str(uuid.uuid4()) if USE_SINGLE_SESSION else None

    for q in QUESTIONS:
        session_id = global_session or str(uuid.uuid4())
        print(f"→ Asking (session {session_id[:8]}): {q}")

        result = call_chat_api(q, session_id=session_id, k=TOP_K)

        if "error" in result:
            print(f"  ! Error: {result['error']}")
            rows.append({
                "session_id": session_id,
                "k": TOP_K,
                "question": q,
                "answer": "",
                "error": result["error"]
            })
            continue

        answer = result.get("answer", "")
        print(f"  ✓ Got answer ({len(answer)} chars)")

        rows.append({
            "session_id": session_id,
            "k": TOP_K,
            "question": q,
            "answer": answer,
            "error": ""
        })

    # Save CSV (simple schema to match /chat output)
    fieldnames = ["session_id", "k", "question", "answer", "error"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
