# merge.py
# -*- coding: utf-8 -*-

import json
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional


# ----------------------------
# Patterns / regex
# ----------------------------
WS_RE = re.compile(r"\s+")
ANSWER_PAT = re.compile(r"^\s*(answer|答案)\s*[:：]", re.IGNORECASE)

# edus dump header
HDR_RE = re.compile(
    r"^---\s*Pred EDUs\s*\(id=(\d+),\s*domain=([a-zA-Z_]+),\s*n=(\d+)\)\s*---\s*$"
)

# edu line: "000     [0,32]  text..."
EDU_LINE_RE = re.compile(r"^\s*(\d+)\s+\[(\d+),(\d+)\]\s*(.*)\s*$")

# option label-only (A. / A / (A) / A )
OPTION_LABEL_ONLY = re.compile(r"^\s*\(?[A-D]\)?\.?\s*$", re.IGNORECASE)

# attribution label-only: "Scientists:" "Critic:" etc.
ATTR_LABEL_ONLY = re.compile(r"^\s*[A-Za-z][A-Za-z \-]{0,30}:\s*$")

# continuation starts
CONT_START = re.compile(
    r"^\s*(that|which|who|whom|whose|as|because|thereby|rather than|in order to|to)\b",
    re.IGNORECASE
)

# discourse / inference cue (often split as a short EDU)
CUE_PREFIX = re.compile(
    r"^\s*("
    r"From\b|According to\b|Since\b|However\b|Therefore\b|Thus\b|Hence\b|"
    r"This implies\b|This means\b|We can conclude\b|which means\b|which implies\b|which states\b|"
    r"In option\b|Option\s+[A-D]\b|The premise\b|The fact\b|This information\b|The original sentence\b"
    r")",
    re.IGNORECASE
)

# parenthetical-only segment
PAREN_ONLY = re.compile(r"^\s*\(.*\)\s*$")


# ----------------------------
# Utility
# ----------------------------
def norm(s: str) -> str:
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
            ctx = ref[max(0, ptr - 80): min(len(ref), ptr + 180)]
            return None, (
                f"Align failed at seg {i}: '{seg_strip[:80]}' ptr={ptr} ctx='{norm(ctx)}'"
            )

        start = ptr + m.start()
        end = ptr + m.end()

        if start < ptr:
            return None, f"Non-monotonic align at seg {i}: start={start} < ptr={ptr}"

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
    """
    q = (question or "").strip()
    c = (cot or "").strip()
    full = (q + joiner + c).strip()

    _, err_full = align_boundaries_in_original_coords(full, gold_adus)
    if err_full is None:
        return full, "full", None

    _, err_cot = align_boundaries_in_original_coords(c, gold_adus)
    if err_cot is None:
        return c, "cot", None

    _, err_q = align_boundaries_in_original_coords(q, gold_adus)
    if err_q is None:
        return q, "question", None

    return None, None, f"full_err={err_full} | cot_err={err_cot} | question_err={err_q}"


# ----------------------------
# Parse your edus dump (TEXT) -> dict[id] = {"domain":..., "edus":[{start,end,text},...]}
# ----------------------------
def read_edus_dump(path: str) -> Dict[int, Dict[str, Any]]:
    cases: Dict[int, Dict[str, Any]] = {}
    cur_id: Optional[int] = None
    cur_domain: Optional[str] = None

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            m = HDR_RE.match(line)
            if m:
                cur_id = int(m.group(1))
                cur_domain = m.group(2)
                cases[cur_id] = {"id": cur_id, "domain": cur_domain, "edus": []}
                continue

            if cur_id is None:
                # ignore preface lines if any
                continue

            m2 = EDU_LINE_RE.match(line)
            if not m2:
                # ignore blank or unexpected lines inside a block
                continue

            idx = int(m2.group(1))
            s = int(m2.group(2))
            e = int(m2.group(3))
            txt = m2.group(4) or ""

            cases[cur_id]["edus"].append({"idx": idx, "start": s, "end": e, "text": txt})

    # sort each case by start
    for cid in list(cases.keys()):
        cases[cid]["edus"].sort(key=lambda x: (x["start"], x["end"], x["idx"]))
    return cases


# ----------------------------
# Merge rules
# ----------------------------
def is_sentence_terminal(t: str) -> bool:
    t = (t or "").strip()
    return bool(re.search(r"[.?!;]\s*$", t))

BROKEN_LEFT_PAREN = re.compile(r"[:\[\(]\s*$")
ENUM_START = re.compile(r"^\s*\(?\d+[\)\.]")

def should_merge(prev_text: str, next_text: str) -> bool:
    p = (prev_text or "").strip()
    n = (next_text or "").strip()

    if not n:
        return False

    # ---------- B1: broken left parenthesis => forbid merge ----------
    if BROKEN_LEFT_PAREN.search(p) and ENUM_START.match(n):
        return False

    # ---------- R1: option label-only ----------
    if OPTION_LABEL_ONLY.match(p):
        return True

    # ---------- R2: attribution label-only ----------
    if ATTR_LABEL_ONLY.match(p):
        return True

    # ---------- R3: discourse cue prefix ----------
    if CUE_PREFIX.match(p) and len(norm(p)) <= 10:
        return True

    # ---------- R4: unfinished sentence chaining ----------
    if not is_sentence_terminal(p):
        if p.endswith(","):
            return True
        if re.match(r"^\s*[a-z]", n):
            return True

    return False



def merge_edus_to_adus(edus: List[Dict[str, Any]], eval_text: str) -> List[Dict[str, Any]]:
    """
    Merge consecutive EDUs (with start/end) into merged ADUs.
    Returned spans are in original eval_text offsets.
    """
    if not edus:
        return []

    # Prefer using slice from eval_text to avoid spacing artifacts in dump
    def edu_text(x):  # safe
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


def merged_boundaries(merged_adus: List[Dict[str, Any]], final_end: int) -> set:
    ends = {x["end"] for x in merged_adus if isinstance(x.get("end"), int)}
    ends.discard(final_end)
    return ends


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="DATA.json", help="Gold dataset JSON (list).")
    ap.add_argument("--edus_dump", default="edus.json", help="Your EDU dump text file (the one you pasted).")
    ap.add_argument("--joiner", default="\n\n", help="Question + joiner + CoT")
    ap.add_argument("--max_items", type=int, default=0, help="0 means all")
    ap.add_argument("--print_cases", action="store_true", help="Print per-case PRF1")
    ap.add_argument("--print_merged", action="store_true", help="Print merged ADUs (per case, line by line)")
    ap.add_argument("--dump_scores", default="merge_case_scores.jsonl")
    ap.add_argument("--dump_merged", default="merged_adus.jsonl")
    ap.add_argument("--dump_failures", default="merge_failures.jsonl")
    args = ap.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_items and args.max_items > 0:
        data = data[:args.max_items]

    pred_map = read_edus_dump(args.edus_dump)  # id -> {domain, edus}

    failures = []
    scores = []
    merged_dump = []

    micro_tp = micro_fp = micro_fn = 0
    macro_list: List[Tuple[float, float, float]] = []

    domain_counter = {"full": 0, "cot": 0, "question": 0}

    for item in data:
        sid = item.get("id")
        q = item.get("Question", "")
        cot = item.get("CoT", "")
        gold_adus = item.get("adus_text", None)

        if not isinstance(gold_adus, list):
            failures.append({"id": sid, "reason": "missing_or_invalid_adus_text"})
            continue

        # strip trailing answer pseudo-ADU if present
        gold_adus = strip_trailing_answer(gold_adus)

        eval_text, domain, derr = choose_eval_text(q, cot, args.joiner, gold_adus)
        if eval_text is None:
            failures.append({"id": sid, "reason": "gold_alignment_failed_all_domains", "detail": derr})
            continue
        domain_counter[domain] += 1

        gold_b, gold_err = align_boundaries_in_original_coords(eval_text, gold_adus)
        if gold_err or gold_b is None:
            failures.append({"id": sid, "reason": "gold_alignment_failed", "domain": domain, "gold_err": gold_err})
            continue

        # get predicted EDUs from dump
        if sid not in pred_map:
            failures.append({"id": sid, "reason": "missing_pred_edus_in_dump"})
            continue
        pred_edus = pred_map[sid]["edus"]

        # sanity check: spans within eval_text
        max_end = max((e["end"] for e in pred_edus), default=0)
        if max_end > len(eval_text):
            failures.append({
                "id": sid,
                "reason": "pred_span_out_of_range",
                "max_end": max_end,
                "eval_text_len": len(eval_text),
                "hint": "This usually means the EDUs were dumped from a different eval_text (e.g., different joiner)."
            })
            continue

        # merge
        merged_adus = merge_edus_to_adus(pred_edus, eval_text)
        pred_b = merged_boundaries(merged_adus, final_end=len(eval_text))

        p, r, f1, tp, fp, fn = boundary_prf(pred_b, gold_b)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        macro_list.append((p, r, f1))

        row = {
            "id": sid,
            "domain": domain,
            "P": p, "R": r, "F1": f1,
            "TP": tp, "FP": fp, "FN": fn,
            "n_gold_adus": len(gold_adus),
            "n_pred_edus": len(pred_edus),
            "n_merged_adus": len(merged_adus),
            "n_gold_boundaries": len(gold_b),
            "n_pred_boundaries": len(pred_b),
        }
        scores.append(row)

        if args.print_cases:
            print(f"[id={sid} domain={domain}] P={p:.4f} R={r:.4f} F1={f1:.4f} | "
                  f"goldADU={len(gold_adus)} predEDU={len(pred_edus)} mergedADU={len(merged_adus)}")

        if args.print_merged:
            print(f"\n--- MERGED ADUs (id={sid}, domain={domain}, n={len(merged_adus)}) ---")
            for i, a in enumerate(merged_adus):
                print(f"{i:03d}\t[{a['start']},{a['end']}]\t{norm(a['text'])}")

        merged_dump.append({
            "id": sid,
            "domain": domain,
            "merged_adus": [
                {"start": x["start"], "end": x["end"], "n_edus": x["n_edus"], "text": x["text"]}
                for x in merged_adus
            ]
        })

    # aggregate
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    if macro_list:
        macro_p = sum(x[0] for x in macro_list) / len(macro_list)
        macro_r = sum(x[1] for x in macro_list) / len(macro_list)
        macro_f1 = sum(x[2] for x in macro_list) / len(macro_list)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    print("\n========== EDU Dump -> Merge -> Boundary Evaluation ==========")
    print(f"Aligned OK cases: {len(macro_list)}")
    print(f"Failed cases: {len(failures)}")
    print(f"Gold domain usage: {domain_counter}\n")

    print("[MICRO]")
    print(f"  TP={micro_tp} FP={micro_fp} FN={micro_fn}")
    print(f"  Precision={micro_p:.4f} Recall={micro_r:.4f} F1={micro_f1:.4f}\n")

    print("[MACRO]")
    print(f"  Precision={macro_p:.4f} Recall={macro_r:.4f} F1={macro_f1:.4f}")

    # dumps
    if args.dump_scores:
        with open(args.dump_scores, "w", encoding="utf-8") as wf:
            for x in scores:
                wf.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"\nPer-case scores dumped to: {args.dump_scores}")

    if args.dump_merged:
        with open(args.dump_merged, "w", encoding="utf-8") as wf:
            for x in merged_dump:
                wf.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Merged ADUs dumped to: {args.dump_merged}")

    if args.dump_failures:
        with open(args.dump_failures, "w", encoding="utf-8") as wf:
            for x in failures:
                wf.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"Failures dumped to: {args.dump_failures}")


if __name__ == "__main__":
    main()
