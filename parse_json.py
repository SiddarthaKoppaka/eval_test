import requests, json

def post_stream_text_then_metadata(
    url: str,
    payload: dict,
    *,
    headers: dict | None = None,
    timeout: int = 600,
    metadata_must_start_on_newline: bool = True,
):
    """
    Sends JSON payload to a streaming endpoint and reads the response incrementally.
    Collects all text until the first complete top-level JSON object is encountered.
    Returns (text, metadata) where metadata is a dict (if JSON parsed) or a raw string.

    If metadata_must_start_on_newline=True, the parser only treats a '{' that appears
    at the start of the stream or immediately after a newline as the metadata start.
    """
    with requests.post(url, json=payload, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()

        text_buf = []        # all text before metadata
        in_meta = False
        meta_buf = []        # chars of the metadata JSON
        depth = 0
        in_str = False
        esc = False
        prev_char = "\n"     # treat start as if preceded by newline

        # We build text in a single buffer until we see metadata.
        # Once we enter metadata mode, we track JSON strings/escapes/braces until depth returns to 0.
        for chunk in r.iter_content(chunk_size=8192, decode_unicode=True):
            if not chunk:
                continue

            for ch in chunk:
                if not in_meta:
                    is_meta_start = (ch == "{") and (
                        not metadata_must_start_on_newline or prev_char == "\n"
                    )
                    if is_meta_start:
                        # switch to metadata mode; do NOT include '{' in text
                        in_meta = True
                        meta_buf.append("{")
                        depth = 1
                        in_str = False
                        esc = False
                    else:
                        text_buf.append(ch)
                else:
                    meta_buf.append(ch)
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                # Done: first full JSON object captured
                                text = "".join(text_buf)
                                meta_str = "".join(meta_buf).strip()
                                try:
                                    meta = json.loads(meta_str)
                                except Exception:
                                    meta = meta_str
                                return text, meta
                prev_char = ch

        # Stream ended without a complete metadata object
        return "".join(text_buf), None
