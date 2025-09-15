# prep_insurance_qna.py
import csv
import json
import os
from typing import List, Dict, Any
import requests

BASE_URL = os.getenv("RAG_API_URL", "http://localhost:5000/ask")
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
OUT_CSV = os.getenv("OUT_CSV", "insurance_qna.csv")

# Mostly insurance policy questions
QUESTIONS: List[str] = [
    "What is the deductible for the standard auto policy and how is it applied per claim?",
    "Does the homeowners policy cover water damage from a burst pipe? Are there exclusions?",
    "Is accidental damage to a mobile phone covered under my renters insurance or do I need a rider?",
    "What are the liability coverage limits included in the default auto policy?",
    "How do I file a claim and what documents are required for health insurance reimbursement?",
    "Does the travel insurance include emergency medical evacuation and repatriation benefits?",
    "Are pre-existing medical conditions covered after a waiting period? If so, how long?",
    "Does the policy cover out-of-network providers, and what is the coinsurance rate?",
    "Is there coverage for temporary rental car while my vehicle is being repaired after an accident?",
    "What perils are excluded under the basic homeowners policy (e.g., flood, earthquake)?",
    "Is maternity coverage included and what are the waiting periods or sub-limits?",
    "How does no-claim bonus (NCB) work, and can it be transferred to a new policy?",
    "Can I add a jewelry or valuables rider, and what appraisal is required?",
    "What is the grace period for premium payments and is there a policy reinstatement window?",
    "How are premiums recalculated after an at-fault auto accident?",
    "Does the life insurance policy allow conversion from term to whole life? Any deadlines?",
    "How do I change beneficiaries on my life insurance and when does it take effect?",
    "Are mental health services covered on par with medical/surgical benefits?",
    "Does dental insurance cover implants or only crowns/bridges, and what are waiting periods?",
    "Is flood damage covered under this policy or does it require a separate flood policy?",
    "Are business-use activities at home covered or excluded under homeowners insurance?",
    "What is the maximum coverage for personal property off-premises (e.g., items in a car)?",
    "Does the travel policy cover trip cancellation due to illness and what proof is needed?",
    "Is roadside assistance included in the auto policy or sold as an add-on?",
    "Are alternative therapies (e.g., chiropractic, acupuncture) covered under health insurance?",
]

def call_api(question: str, k: int = TOP_K) -> Dict[str, Any]:
    payload = {"input": question, "k": k}
    try:
        r = requests.post(BASE_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()  # expects: {"answer": str, "prompt": str, "sources": [...]}
    except Exception as e:
        return {"error": str(e)}

def main():
    rows = []
    for q in QUESTIONS:
        print(f"→ Asking: {q}")
        result = call_api(q)
        if "error" in result:
            print(f"  ! Error: {result['error']}")
            rows.append({
                "question": q,
                "answer": "",
                "prompt": "",
                "sources": "",
                "error": result["error"]
            })
            continue

        answer = result.get("answer", "")
        prompt = result.get("prompt", "")
        sources = result.get("sources", [])
        sources_json = json.dumps(sources, ensure_ascii=False)

        rows.append({
            "question": q,
            "answer": answer,
            "prompt": prompt,
            "sources": sources_json,
            "error": ""
        })
        print(f"  ✓ Got answer ({len(answer)} chars)")

    # Write CSV
    fieldnames = ["question", "answer", "prompt", "sources", "error"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
