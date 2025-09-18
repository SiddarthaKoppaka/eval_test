import requests
import json

def get_answers(payload, api_url, headers=None, timeout=60):
    """
    Stream a response and return only generated text.
    Handles:
      - raw text lines
      - SSE lines starting with 'data:'
      - JSON chunks: {'text': ...}, {'delta': {'text': ...}},
        OpenAI-like {'choices':[{'delta':{'content': ...}}]}
        Anthropic-like {'type':'content_block_delta','delta':{'text': ...}}
    Reads the body ONCE to avoid 'content already consumed' errors.
    """
    base_headers = {
        "Accept": "text/event-stream, application/json;q=0.9, */*;q=0.8",
        "Content-Type": "application/json",
        # Optional: nudge servers to stream
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    if headers:
        base_headers.update(headers)

    collected = []
    raw_lines = []

    with requests.post(api_url, json=payload, headers=base_headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue

            line = raw.strip()
            if not line:
                continue

            raw_lines.append(line)

            # SSE "data:" prefix
            if line.startswith("data:"):
                line = line[5:].lstrip()
                if not line:
                    continue
                # Some streams send [DONE]
                if line == "[DONE]":
                    break

            # Try JSON chunk
            if line.startswith("{") or line.startswith("["):
                try:
                    obj = json.loads(line)
                except Exception:
                    # Not valid JSON -> treat as plain text
                    collected.append(line)
                    continue

                # Common shapes:

                # 1) Plain {"text": "..."}
                if isinstance(obj, dict) and "text" in obj:
                    if obj["text"]:
                        collected.append(str(obj["text"]))
                    continue

                # 2) Plain {"delta":{"text":"..."}}
                if isinstance(obj, dict) and isinstance(obj.get("delta"), dict) and "text" in obj["delta"]:
                    if obj["delta"]["text"]:
                        collected.append(str(obj["delta"]["text"]))
                    continue

                # 3) OpenAI-like {"choices":[{"delta":{"content":"..."}}], ...}
                if isinstance(obj, dict) and isinstance(obj.get("choices"), list):
                    for ch in obj["choices"]:
                        delta = ch.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            collected.append(str(content))
                    continue

                # 4) Anthropic-like chunk
                # {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}, ...}
                if (
                    isinstance(obj, dict)
                    and obj.get("type") == "content_block_delta"
                    and isinstance(obj.get("delta"), dict)
                    and "text" in obj["delta"]
                ):
                    collected.append(str(obj["delta"]["text"]))
                    continue

                # If JSON but not a known text-shape, ignore silently
                continue

            # Fallback: treat as plain text line
            collected.append(line)

    # Primary return: aggregated deltas/plain text
    if collected:
        return "".join(collected)

    # Fallback: if nothing recognized, return the raw body we saw
    # (No second read attempt—use the buffered lines)
    return "\n".join(raw_lines)
