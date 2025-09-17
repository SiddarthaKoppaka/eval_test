import requests, json

def read_text_then_metadata_json(url, headers=None):
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()

        text_parts = []
        buf = ""           # general buffer while scanning for "\n{"
        in_meta = False
        meta = []
        depth = 0
        in_str = False
        esc = False

        def flush_text(s):
            if s:
                text_parts.append(s)

        for chunk in r.iter_content(chunk_size=8192, decode_unicode=True):
            if not chunk:
                continue
            i = 0
            while i < len(chunk):
                ch = chunk[i]

                if not in_meta:
                    buf += ch
                    # start metadata when we see '{' at start or after a newline
                    if ch == '{' and (len(buf) == 1 or buf[-2] == '\n'):
                        # Cut text before this '{'
                        start_of_meta_idx = len(buf) - 1
                        flush_text(buf[:start_of_meta_idx])
                        buf = ""  # reset
                        in_meta = True
                        depth = 1
                        in_str = False
                        esc = False
                        meta = ['{']
                    # else keep buffering text
                else:
                    meta.append(ch)
                    # JSON string/escape tracking
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == '\\':
                            esc = True
                        elif ch == '"':
                            in_str = False
                    else:
                        if ch == '"':
                            in_str = True
                        elif ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                # Done reading metadata JSON
                                text = "".join(text_parts)
                                meta_str = "".join(meta).strip()
                                try:
                                    return text, json.loads(meta_str)
                                except json.JSONDecodeError:
                                    # Fallback: return raw string if not valid JSON
                                    return text, meta_str
                i += 1

        # If we ended without seeing metadata:
        flush_text(buf)
        return "".join(text_parts), None
