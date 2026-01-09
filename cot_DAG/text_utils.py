import re
from typing import List, Tuple

WS_RE = re.compile(r"\s+")
_SENT_END_RE = re.compile(r"([\.!?;。！？；]+)")

def normalize_ws(s: str) -> str:
    return WS_RE.sub(" ", (s or "")).strip()

def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    t = normalize_ws(text)
    if not t:
        return []

    parts = _SENT_END_RE.split(t)
    sents = []
    cursor = 0
    buf = ""
    buf_start = 0

    def flush():
        nonlocal buf, buf_start
        sent = buf.strip()
        if sent:
            start = buf_start
            end = start + len(buf)
            lstrip = len(buf) - len(buf.lstrip())
            rstrip = len(buf) - len(buf.rstrip())
            start += lstrip
            end -= rstrip
            sents.append((start, end, t[start:end]))
        buf = ""

    for i in range(0, len(parts), 2):
        chunk = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        piece = chunk + punct
        if not buf:
            buf_start = cursor
        buf += piece
        cursor += len(piece)
        if punct:
            flush()

    if buf.strip():
        flush()

    if not sents:
        return [(0, len(t), t)]
    return sents
