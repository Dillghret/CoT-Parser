from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class FewShot:
    sentence: str
    adus_text: List[str]


def load_fewshots(path: str) -> List[FewShot]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, list):
        raise ValueError(f"fewshot file must be a list, got {type(obj)}")

    out: List[FewShot] = []
    for i, x in enumerate(obj):
        if not isinstance(x, dict):
            raise ValueError(f"fewshot[{i}] must be a dict, got {type(x)}")
        sent = x.get("sentence", "")
        adus = x.get("adus_text", [])
        if not isinstance(sent, str):
            raise ValueError(f"fewshot[{i}].sentence must be str")
        if not isinstance(adus, list) or not all(isinstance(a, str) for a in adus):
            raise ValueError(f"fewshot[{i}].adus_text must be list[str]")
        out.append(FewShot(sentence=sent, adus_text=[a for a in adus if a.strip()]))

    return out


def render_fewshot_block(shots: Sequence[FewShot]) -> str:
    blocks = []
    for i, s in enumerate(shots, start=1):
        blocks.append(f"[Example {i}]\\nSentence: {s.sentence}\\nADUs: {json.dumps(s.adus_text, ensure_ascii=False)}")
    return "\\n\\n".join(blocks)
