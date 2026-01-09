from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .fewshots import load_fewshots
from .acs_llm import segment_text_llm
from .acs_unirst import segment_text_unirst
from .aric_llm import infer_aric
from .llm_client import DEFAULT_BASE_URL, resolve_api_key

DEFAULT_FEWSHOTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "fewshots", "default.json")


def infer_from_text(
    text: str,
    *,
    ac_method: str = "llm",
    ac_model: str = "qwen3-max",
    aric_model: str = "qwen3-max",
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    fewshot_json: Optional[str] = None,
    unirst_hf_model_name: str = "tchewik/isanlp_rst_v3",
    unirst_version: str = "unirst",
    unirst_relinventory: str = "eng.erst.gum",
    unirst_cuda_device: int = -1,
    save_debug: bool = False,
) -> Dict[str, Any]:
    """
    Run AC segmentation + ARIC inference on a single text.
    """
    key = resolve_api_key(api_key)

    if ac_method.lower() in "llm":
        shots_path = fewshot_json or os.path.abspath(DEFAULT_FEWSHOTS_PATH)
        fewshots = load_fewshots(shots_path)
        seg = segment_text_llm(
            text,
            base_url=base_url,
            api_key=key,
            model=ac_model,
            fewshots=fewshots,
        )
    elif ac_method.lower() in "unirst":
        seg = segment_text_unirst(
            text,
            hf_model_name=unirst_hf_model_name,
            hf_model_version=unirst_version,
            relinventory=unirst_relinventory,
            cuda_device=unirst_cuda_device,
            do_merge=True,
        )
    else:
        raise ValueError(f"Unknown ac_method: {ac_method}")

    adus_text = seg.get("adus_text", [])
    if not isinstance(adus_text, list):
        adus_text = []

    aric = infer_aric(adus_text, base_url=base_url, api_key=key, model=aric_model)

    out = {
        "text": text,
        "ac": {
            "method": seg.get("method", ac_method),
            "adus_text": adus_text,
        },
        "aric": {
            "model": aric_model,
            "edges": aric.get("edges", []),
        },
    }
    if save_debug:
        out["debug"] = {"ac": seg, "aric_raw": aric.get("raw", {})}
    return out
