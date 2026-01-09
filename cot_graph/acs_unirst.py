from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .merge_rules import merge_edus_to_adus

def _require_unirst():
    try:
        from isanlp_rst.parser import Parser
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "UniRST dependencies are missing. Install:\n"
            "  pip uninstall isanlp -y && pip install git+https://github.com/iinemo/isanlp.git\n"
            "  pip install isanlp_rst\n"
            f"Original import error: {e}"
        )

def extract_pred_leaves(res: Dict[str, Any]) -> List[Tuple[int, int]]:
    if "rst" not in res or not res["rst"]:
        raise KeyError("Missing 'rst' in parser output.")
    root = res["rst"][0]
    spans: List[Tuple[int, int]] = []

    stack = [root]
    while stack:
        node = stack.pop()
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)

        if left is None and right is None:
            s = getattr(node, "start", None)
            e = getattr(node, "end", None)
            if isinstance(s, int) and isinstance(e, int) and e > s:
                spans.append((s, e))
        else:
            if right is not None:
                stack.append(right)
            if left is not None:
                stack.append(left)

    return sorted(set(spans), key=lambda x: (x[0], x[1]))


def segment_text_unirst(
    text: str,
    *,
    hf_model_name: str = "tchewik/isanlp_rst_v3",
    hf_model_version: str = "unirst",
    relinventory: str = "eng.erst.gum",
    cuda_device: int = -1,
    do_merge: bool = True,
) -> Dict[str, Any]:
    """
    Segment a single text into ADUs using UniRST EDUs + merge rules.
    """
    _require_unirst()
    from isanlp_rst.parser import Parser

    t = text or ""
    if not t.strip():
        return {"method": "unirst", "edus": [], "merged_adus": [], "adus_text": []}

    parser = Parser(
        hf_model_name=hf_model_name,
        hf_model_version=hf_model_version,
        cuda_device=cuda_device,
        relinventory=relinventory,
    )

    res = parser(t)
    spans = extract_pred_leaves(res)
    edus = [{"start": s, "end": e, "text": t[s:e]} for (s, e) in spans]

    if do_merge:
        merged = merge_edus_to_adus(edus, t)
        adus_text = [m["text"].strip() for m in merged if isinstance(m.get("text"), str) and m["text"].strip()]
        return {"method": "unirst", "edus": edus, "merged_adus": merged, "adus_text": adus_text}

    adus_text = [x["text"].strip() for x in edus if x.get("text", "").strip()]
    return {"method": "unirst", "edus": edus, "merged_adus": [], "adus_text": adus_text}
