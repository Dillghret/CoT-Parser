# eval_unirst_boundary_verbose.py
# -*- coding: utf-8 -*-

import json
import re
import argparse
from typing import List, Tuple, Dict, Any, Optional

from isanlp_rst.parser import Parser

WS_RE = re.compile(r"\s+")
ANSWER_PAT = re.compile(r"^\s*(answer|答案)\s*[:：]", re.IGNORECASE)


def normalize_text(s: str) -> str:
    return WS_RE.sub(" ", (s or "")).strip()


def strip_trailing_answer(adus: List[str]) -> List[str]:
    """Remove trailing 'answer: X' pseudo-ADU if present (only last one)."""
    if not adus:
        return adus
    last = adus[-1]
    if isinstance(last, str) and ANSWER_PAT.match(last.strip()):
        return adus[:-1]
    return adus


def boundary_prf(pred_b: set, gold_b: set) -> Tuple[float, float, float, int, int, int]:
    tp = len(pred_b & gold_b)
    fp = len(pred_b - gold_b)
    fn = len(gold_b - pred_b)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


def align_boundaries_in_original_coords(ref: str, segments: List[str]) -> Tuple[Optional[set], Optional[str]]:
    """
    Align segments to ref in ORIGINAL coordinates and return original end offsets.
    Whitespace-tolerant: whitespace in segment matches \\s+ in ref.
    """
    boundaries = set()
    ptr = 0

    for i, seg in enumerate(segments):
        if not isinstance(seg, str):
            return None, f"Non-string segment at {i}: type={type(seg)}"

        seg_strip = seg.strip()
        if not seg_strip:
            continue

        parts = WS_RE.split(seg_strip)
        parts = [re.escape(p) for p in parts if p]
        if not parts:
            continue
        pattern = r"\s+".join(parts)

        m = re.search(pattern, ref[ptr:], flags=re.DOTALL)
        if not m:
            context = ref[max(0, ptr - 60): min(len(ref), ptr + 120)]
            return None, (
                f"Original align failed at segment {i}: '{seg_strip[:80]}' "
                f"ptr={ptr} ctx='{normalize_text(context)}'"
            )

        start = ptr + m.start()
        end = ptr + m.end()

        if start < ptr:
            return None, f"Non-monotonic original align at segment {i}: start={start} < ptr={ptr}"

        ptr = end
        boundaries.add(end)

    boundaries.discard(len(ref))
    return boundaries, None


def choose_eval_text(question: str, cot: str, joiner: str, gold_adus: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Decide which text domain gold_adus belongs to by attempting alignment:
      1) full = Question + joiner + CoT
      2) CoT only
      3) Question only
    Returns (eval_text, domain, err_if_all_failed)
    """
    q = (question or "").strip()
    c = (cot or "").strip()
    full = (q + joiner + c).strip()

    b, err = align_boundaries_in_original_coords(full, gold_adus)
    if err is None:
        return full, "full", None

    b, err2 = align_boundaries_in_original_coords(c, gold_adus)
    if err2 is None:
        return c, "cot", None

    b, err3 = align_boundaries_in_original_coords(q, gold_adus)
    if err3 is None:
        return q, "question", None

    return None, None, f"full_err={err} | cot_err={err2} | question_err={err3}"


def extract_pred_leaves(res: Dict[str, Any]) -> List[Tuple[int, int]]:
    """
    Return leaf spans as (start, end) from UniRST tree.
    """
    if "rst" not in res or not res["rst"]:
        raise KeyError(f"Missing 'rst' in parser output. res.keys()={list(res.keys())}")

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
            # left-to-right
            if right is not None:
                stack.append(right)
            if left is not None:
                stack.append(left)

    # sort, unique
    spans = sorted(set(spans), key=lambda x: (x[0], x[1]))
    return spans


def spans_to_boundaries(spans: List[Tuple[int, int]], final_end: Optional[int] = None) -> List[int]:
    ends = sorted({e for (_, e) in spans})
    if final_end is not None and final_end in ends:
        ends.remove(final_end)
    return ends


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="DATA.json", help="Path to your JSON file (list of dicts).")
    ap.add_argument("--cuda_device", type=int, default=0, help="GPU id; use -1 for CPU.")
    ap.add_argument("--hf_model_name", default="tchewik/isanlp_rst_v3")
    ap.add_argument("--version", default="unirst", choices=["gumrrg", "rstdt", "rstreebank", "unirst"])
    ap.add_argument("--relinventory", default="eng.erst.gum")
    ap.add_argument("--joiner", default="\n\n", help="How to concatenate Question and CoT for full-text eval.")
    ap.add_argument("--max_items", type=int, default=0, help="0 means all; otherwise evaluate first N.")
    ap.add_argument("--dump_failures", default="align_failures.jsonl", help="Write failures here (jsonl).")
    ap.add_argument("--dump_cases", default="case_scores.jsonl", help="Write per-case scores here (jsonl).")
    ap.add_argument("--dump_edus", default="case_edus.jsonl", help="Write per-case EDUs here (jsonl).")
    ap.add_argument("--print_cases", action="store_true", help="Print per-case PRF1 to stdout.")
    ap.add_argument("--print_edus", action="store_true", help="Print EDUs for each aligned case to stdout (may be long).")
    args = ap.parse_args()

    parser = Parser(
        hf_model_name=args.hf_model_name,
        hf_model_version=args.version,
        cuda_device=args.cuda_device,
        relinventory=args.relinventory
    )

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items.")

    if args.max_items and args.max_items > 0:
        data = data[:args.max_items]

    micro_tp = micro_fp = micro_fn = 0
    macro_scores: List[Tuple[float, float, float]] = []
    failures: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    edu_rows: List[Dict[str, Any]] = []
    domain_counter = {"full": 0, "cot": 0, "question": 0}

    for item in data:
        sid = item.get("id", None)
        q = item.get("Question", "")
        cot = item.get("CoT", "")
        gold_adus = item.get("adus_text", None)

        if gold_adus is None:
            failures.append({"id": sid, "reason": "missing adus_text"})
            continue
        if not isinstance(gold_adus, list):
            failures.append({"id": sid, "reason": "adus_text is not a list"})
            continue

        gold_adus = strip_trailing_answer(gold_adus)

        eval_text, domain, domain_err = choose_eval_text(q, cot, args.joiner, gold_adus)
        if eval_text is None:
            failures.append({
                "id": sid,
                "reason": "gold_alignment_failed_on_all_domains",
                "detail": domain_err,
                "gold_first3": [normalize_text(x) for x in gold_adus[:3]],
                "q_head": normalize_text(q)[:200],
                "cot_head": normalize_text(cot)[:200],
            })
            continue
        domain_counter[domain] += 1

        # gold boundaries in original coords
        gold_b, gold_err = align_boundaries_in_original_coords(eval_text, gold_adus)
        if gold_err or gold_b is None:
            failures.append({
                "id": sid,
                "reason": "gold_alignment_failed",
                "domain": domain,
                "gold_err": gold_err,
                "eval_text_head": normalize_text(eval_text)[:300],
                "gold_first3": [normalize_text(x) for x in gold_adus[:3]],
            })
            continue

        # pred boundaries + EDUs
        try:
            res = parser(eval_text)
            spans = extract_pred_leaves(res)  # [(start,end),...]
            # final end from root if available
            root = res["rst"][0]
            final_end = getattr(root, "end", None) if root is not None else None
            pred_ends = spans_to_boundaries(spans, final_end=final_end)
            pred_b = set(pred_ends)
        except Exception as e:
            failures.append({"id": sid, "reason": f"parser/pred_extract_failed: {repr(e)}"})
            continue

        p, r, f1, tp, fp, fn = boundary_prf(pred_b, gold_b)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        macro_scores.append((p, r, f1))

        case_row = {
            "id": sid,
            "domain": domain,
            "P": p,
            "R": r,
            "F1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "n_gold_boundaries": len(gold_b),
            "n_pred_boundaries": len(pred_b),
            "n_gold_adus": len(gold_adus),
            "n_pred_edus": len(spans),
        }
        case_rows.append(case_row)

        # dump EDUs for inspection
        edu_texts = []
        for (s, e) in spans:
            edu_texts.append({
                "start": s,
                "end": e,
                "text": eval_text[s:e],
            })
        edu_rows.append({
            "id": sid,
            "domain": domain,
            "eval_text_head": normalize_text(eval_text)[:300],
            "pred_edus": edu_texts
        })

        if args.print_cases:
            print(f"[case id={sid} domain={domain}] P={p:.4f} R={r:.4f} F1={f1:.4f} (TP={tp} FP={fp} FN={fn})")

        if args.print_edus:
            print(f"--- EDUs for case id={sid} domain={domain} ---")
            for j, edu in enumerate(edu_texts):
                t = normalize_text(edu["text"])
                print(f"{j:03d} [{edu['start']},{edu['end']}] {t}")
            print("")

    n_ok = len(macro_scores)

    # Micro
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # Macro
    if macro_scores:
        macro_p = sum(x[0] for x in macro_scores) / n_ok
        macro_r = sum(x[1] for x in macro_scores) / n_ok
        macro_f1 = sum(x[2] for x in macro_scores) / n_ok
    else:
        macro_p = macro_r = macro_f1 = 0.0

    print("\n========== UniRST Segmentation Boundary Evaluation ==========")
    print(f"Evaluated items (aligned OK): {n_ok}")
    print(f"Failed items: {len(failures)}")
    print(f"Gold domain usage: {domain_counter}\n")

    print("[MICRO boundary PRF]")
    print(f"  TP={micro_tp} FP={micro_fp} FN={micro_fn}")
    print(f"  Precision={micro_p:.4f} Recall={micro_r:.4f} F1={micro_f1:.4f}\n")

    print("[MACRO boundary PRF]")
    print(f"  Precision={macro_p:.4f} Recall={macro_r:.4f} F1={macro_f1:.4f}")

    # dumps
    if args.dump_failures and failures:
        with open(args.dump_failures, "w", encoding="utf-8") as wf:
            for x in failures:
                wf.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"\nFailures dumped to: {args.dump_failures}")

    if args.dump_cases and case_rows:
        with open(args.dump_cases, "w", encoding="utf-8") as wf:
            for x in case_rows:
                wf.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Per-case scores dumped to: {args.dump_cases}")

    if args.dump_edus and edu_rows:
        with open(args.dump_edus, "w", encoding="utf-8") as wf:
            for x in edu_rows:
                wf.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Per-case EDUs dumped to: {args.dump_edus}")


if __name__ == "__main__":
    main()
