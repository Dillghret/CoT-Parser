from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from cot_graph.pipeline import infer_from_text
from cot_graph.llm_client import DEFAULT_BASE_URL


def build_out_path(*, input_txt: str, out_dir: str, out_name: str, name_template: str,
                   ac_method: str, ac_model: str, aric_model: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_txt))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if out_name:
        filename = out_name
    else:
        tpl = name_template or "{stem}_AC-{ac_method}_ACM-{ac_model}_ARIC-{aric_model}_{ts}.json"
        filename = tpl.format(stem=stem, ac_method=ac_method, ac_model=ac_model, aric_model=aric_model, ts=ts)

    if not filename.lower().endswith(".json"):
        filename += ".json"
    return os.path.join(out_dir, filename)


def main():
    ap = argparse.ArgumentParser(description="Inference")
    ap.add_argument("--input_txt", default="../data/input.txt")
    ap.add_argument("--out_dir", default="../data/outputs")
    ap.add_argument("--out_name", default="")
    ap.add_argument("--name_template", default="")

    ap.add_argument("--ac_method", default="llm", choices=["llm", "unirst"])
    ap.add_argument("--ac_model", default="qwen3-max")
    ap.add_argument("--aric_model", default="qwen3-max")
    ap.add_argument("--base_url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api_key", default="sk-58ffd45bdcb84fbe800067e62783f417")

    ap.add_argument("--fewshot_json", default="")
    ap.add_argument("--save_debug", action="store_true")

    ap.add_argument("--unirst_hf_model_name", default="tchewik/isanlp_rst_v3")
    ap.add_argument("--unirst_version", default="unirst")
    ap.add_argument("--unirst_relinventory", default="eng.erst.gum")
    ap.add_argument("--unirst_cuda_device", type=int, default=-1)

    args = ap.parse_args()

    with open(args.input_txt, "r", encoding="utf-8") as f:
        text = f.read().strip()

    out_path = build_out_path(
        input_txt=args.input_txt,
        out_dir=args.out_dir,
        out_name=args.out_name,
        name_template=args.name_template,
        ac_method=args.ac_method,
        ac_model=args.ac_model,
        aric_model=args.aric_model,
    )

    result = infer_from_text(
        text,
        ac_method=args.ac_method,
        ac_model=args.ac_model,
        aric_model=args.aric_model,
        base_url=args.base_url,
        api_key=(args.api_key or None),
        fewshot_json=(args.fewshot_json or None),
        unirst_hf_model_name=args.unirst_hf_model_name,
        unirst_version=args.unirst_version,
        unirst_relinventory=args.unirst_relinventory,
        unirst_cuda_device=args.unirst_cuda_device,
        save_debug=args.save_debug,
    )

    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(result, wf, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {out_path}")
    print(f"ADUs: {len(result.get('ac', {}).get('adus_text', []))}")
    print(f"Edges: {len(result.get('aric', {}).get('edges', []))}")


if __name__ == "__main__":
    main()
