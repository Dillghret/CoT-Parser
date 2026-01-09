from __future__ import annotations

import re
from typing import Any, Dict, List

WS_RE = re.compile(r"\s+")

OPTION_LABEL_ONLY = re.compile(r"^\s*([A-D])\s*[\)\.\:\-]\s*$", re.I)
ATTR_LABEL_ONLY = re.compile(r"^\s*(premise|claim|conclusion)\s*[\)\.\:\-]?\s*$", re.I)
CUE_PREFIX = re.compile(r"^\s*(because|however|therefore|thus|so|but|and|or)\b", re.I)

BROKEN_LEFT_PAREN = re.compile(r"[:\[\(]\s*$")
ENUM_START = re.compile(r"^\s*\(?\d+[\)\.]")

def norm(s: str) -> str:
    return WS_RE.sub(" ", (s or "")).strip()

def is_sentence_terminal(t: str) -> bool:
    t = (t or "").strip()
    return bool(re.search(r"[.?!;。！？；]\s*$", t))

def should_merge(prev_text: str, next_text: str) -> bool:
    p = (prev_text or "").strip()
    n = (next_text or "").strip()

    if not n:
        return False

    # B1: broken left parenthesis / colon + enumerator => forbid merge
    if BROKEN_LEFT_PAREN.search(p) and ENUM_START.match(n):
        return False

    # R1: option label-only
    if OPTION_LABEL_ONLY.match(p):
        return True

    # R2: attribution label-only
    if ATTR_LABEL_ONLY.match(p):
        return True

    # R3: discourse cue prefix
    if CUE_PREFIX.match(p) and len(norm(p)) <= 10:
        return True

    # R4: unfinished sentence chaining
    if not is_sentence_terminal(p):
        if p.endswith(","):
            return True
        if re.match(r"^\s*[a-z]", n):
            return True

    return False


def merge_edus_to_adus(edus: List[Dict[str, Any]], eval_text: str) -> List[Dict[str, Any]]:
    if not edus:
        return []

    def edu_text(x):
        s, e = x["start"], x["end"]
        if 0 <= s < e <= len(eval_text):
            return eval_text[s:e]
        return x.get("text", "")

    merged = []
    buf_start = edus[0]["start"]
    buf_end = edus[0]["end"]
    buf_text = edu_text(edus[0])
    buf_n = 1

    for i in range(1, len(edus)):
        nxt = edus[i]
        nxt_text = edu_text(nxt)

        if should_merge(buf_text, nxt_text):
            buf_end = nxt["end"]
            buf_text = eval_text[buf_start:buf_end] if (0 <= buf_start < buf_end <= len(eval_text)) else (buf_text + " " + nxt_text)
            buf_n += 1
        else:
            merged.append({"start": buf_start, "end": buf_end, "text": buf_text, "n_edus": buf_n})
            buf_start = nxt["start"]
            buf_end = nxt["end"]
            buf_text = nxt_text
            buf_n = 1

    merged.append({"start": buf_start, "end": buf_end, "text": buf_text, "n_edus": buf_n})
    return merged
