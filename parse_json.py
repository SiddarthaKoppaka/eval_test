import requests
import json

def get_answers(payload, api_url, headers=None, timeout=60):
    """
    POSTs payload to api_url and reads a streaming response.
    Returns only the generated text (metadata is ignored).
    Works with:
      - raw text chunks
      - SSE lines starting with 'data:'
      - JSON chunks like {'text': '...'} or {'delta': {'text':'...'}}
    Falls back to non-stream JSON/text if streaming isn't used.
    """
    base_headers = {
        "Accept": "text/event-stream, application/json;q=0.9, */*;q=0.8",
        "Content-Type": "application/json",
    }
    if headers:
        base_headers.update(headers)

    with requests.post(api_url, json=payload, headers=base_headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        parts = []
        metadata_started = False

        # Stream line-by-line (robust for SSE and newline-delimited JSON)
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue

            # Handle SSE prefix
            if line.startswith("data:"):
                line = line[5:].lstrip()
                if not line:
                    continue

            # If a JSON object/array starts, try to parse it
            if line.startswith("{") or line.startswith("["):
                try:
                    obj = json.loads(line)
                except Exception:
                    # Not valid JSON—treat as plain text
                    if not metadata_started:
                        parts.append(line)
                    continue

                # Treat objects with 'metadata' (or JSON lacking any text) as metadata and stop capturing
                if isinstance(obj, dict):
                    # Common streaming shapes
                    if "text" in obj:
                        parts.append(str(obj["text"]))
                        continue
                    if "delta" in obj and isinstance(obj["delta"], dict) and "text" in obj["delta"]:
                        parts.append(str(obj["delta"]["text"]))
                        continue
                    if "metadata" in obj:
                        metadata_started = True
                        break

                    # If it's JSON but not text/delta, consider it metadata-ish; ignore it
                    continue

                # Arrays usually aren’t text tokens—ignore
                continue

            # Plain text chunk
            if not metadata_started:
                parts.append(line)

        # If we got nothing from the stream, try non-stream body
        if not parts:
            try:
                data = r.json()
                if isinstance(data, dict) and "text" in data:
                    return str(data["text"])
                return r.text
            except Exception:
                return r.text

        return "".join(parts)
