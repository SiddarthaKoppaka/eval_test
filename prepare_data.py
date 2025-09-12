# scripts/build_structured_dataset.py
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# If this import causes circulars, move generate_structured_story to a service module
# e.g., from app.services.story_generator import generate_structured_story
from app.main import generate_structured_story  # <-- uses your existing function

PROMPTS: List[str] = [
    "A lonely lighthouse keeper discovers a mysterious glowing pearl that washes ashore.",
    "A young witch accidentally turns her grumpy cat into a talking, flying broomstick.",
    "An astronaut gets stranded on a planet where the plants communicate through music.",
    "A detective in a steampunk city must solve the theft of a priceless clockwork heart.",
    "Two rival chefs must team up to win a magical cooking competition.",
    "A timid librarian finds a book that allows him to enter into the stories he reads.",
    "A group of kids builds a spaceship out of junk and actually travels to the moon.",
    "A knight who is afraid of dragons is tasked with saving a princess from one.",
    "An ancient robot wakes up in a post-apocalyptic world and tries to find its purpose.",
    "A musician discovers her guitar can control the weather when she plays certain chords.",
]

OUT_DIR = Path("out")
RAW_JSONL = OUT_DIR / "stories.jsonl"
TABLE_CSV = OUT_DIR / "stories.csv"
EVAL_CSV = OUT_DIR / "eval_dataset.csv"


def _as_dict(payload: Any) -> Dict[str, Any]:
    """
    Accepts dict or JSON string; returns dict.
    """
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            # best effort: store as plain text
            return {"story": payload}
    # unexpected types
    return {"story": str(payload)}


def _extract_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize common fields your generator returns.
    Falls back gracefully if keys are missing or named differently.
    """
    # Standard keys you mentioned
    title = obj.get("title") or obj.get("story_title")
    characters = obj.get("characters")
    setting = obj.get("setting")

    # Prefer the long-form narrative as the response/story
    story = obj.get("story") or obj.get("summary") or obj.get("outline") or ""

    # Optional model metadata if your generator includes it
    model_id = obj.get("model_id") or obj.get("model") or os.getenv("BEDROCK_MODEL_ID") or None

    return {
        "title": title,
        "characters": characters,
        "setting": setting,
        "story": story,
        "model_id": model_id,
        # keep the full original object too (to be written in JSONL)
        "_full": obj,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_JSONL.exists():
        RAW_JSONL.unlink()  # start fresh

    rows_for_table: List[Dict[str, Any]] = []
    rows_for_eval: List[Dict[str, Any]] = []

    print("Generating structured stories...")
    for prompt in PROMPTS:
        try:
            raw = generate_structured_story(prompt)
            obj = _as_dict(raw)
            fields = _extract_fields(obj)

            sample_id = uuid.uuid4().hex[:8]
            created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

            # Append to table rows (rich, human-readable)
            rows_for_table.append(
                {
                    "id": sample_id,
                    "created_at": created_at,
                    "question": prompt,
                    "title": fields["title"],
                    "characters": fields["characters"],
                    "setting": fields["setting"],
                    "story": fields["story"],
                    "model_id": fields["model_id"],
                }
            )

            # Append to eval rows (lean, Ragas-ready)
            rows_for_eval.append(
                {
                    "user_input": prompt,
                    "response": fields["story"],   # Ragas "answer" equivalent
                    "reference": "",               # keep blank unless you have gold answers
                }
            )

            # Also write each full payload to JSONL for complete traceability
            with RAW_JSONL.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps({
                    "id": sample_id,
                    "created_at": created_at,
                    "question": prompt,
                    "payload": fields["_full"],
                }, ensure_ascii=False) + "\n")

            print(f"✓ {sample_id} — {prompt[:60]}...")

        except Exception as e:
            print(f"✗ Failed for prompt: {prompt}\n  Error: {e}")

    # Save the human-readable table
    if rows_for_table:
        df = pd.DataFrame(rows_for_table)
        df.to_csv(TABLE_CSV, index=False, encoding="utf-8")
        print(f"\nSaved {len(df)} rows to {TABLE_CSV}")

    # Save the eval dataset
    if rows_for_eval:
        eval_df = pd.DataFrame(rows_for_eval)
        eval_df.to_csv(EVAL_CSV, index=False, encoding="utf-8")
        print(f"Saved evaluation-ready CSV to {EVAL_CSV}")

    if not rows_for_table and not rows_for_eval:
        print("\nNo data generated.")


if __name__ == "__main__":
    main()
