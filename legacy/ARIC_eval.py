
import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import time
import random
from datetime import datetime

import httpx
import certifi

try:
    from openai import OpenAI
    from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
except Exception:
    OpenAI = None


# ===========================
# Label set for ARIC
# ===========================

LABELS = {
    "Support": "ADU A provides justification, evidence, or reasoning that strengthens ADU B.",
    "Attack": "ADU A directly denies, contradicts, or rejects the content or correctness of ADU B.",
    "Restatement": "ADU A rephrases or repeats the content of ADU B without adding new constraints.",
    "Non-logical": "ADU A and ADU B are related in a non-argumentative way (e.g., pure background, meta-comment) and A is not used as a reason to accept/reject B.",
    "No-Relation": "No clear argumentative relation between ADU A and ADU B."
}

LABEL_CANON = {lbl.lower(): lbl for lbl in LABELS.keys()}


# ===========================
# IO helpers
# ===========================

def load_items(path: str):
    """
    Robust loader:
    - If the whole file is a single JSON array/object -> yield from it
    - Else fallback to JSONL (one JSON object per line)
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.lstrip("\ufeff").strip()

    # Try full JSON first
    try:
        obj = json.loads(content)
        if isinstance(obj, list):
            for it in obj:
                if it is not None:
                    yield it
            return
        elif isinstance(obj, dict):
            yield obj
            return
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    line_no = 0
    for line in content.splitlines():
        line_no += 1
        s = line.strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except json.JSONDecodeError as e:
            print(f"[WARN] skip invalid JSON at line {line_no}: {e}")
            continue


def make_item_key(item: Dict[str, Any], fallback_seq: int) -> str:
    """Unique key for resume; prefer id+idx, else fallback to seq."""
    idv = item.get("id")
    idx = item.get("idx")
    if idv is not None and idx is not None:
        return f"id={idv}|idx={idx}"
    if idv is not None:
        return f"id={idv}"
    if idx is not None:
        return f"idx={idx}"
    return f"seq={fallback_seq}"


def load_done_keys(out_jsonl: str) -> set:
    """Scan existing output jsonl to collect processed keys; robust to partial files."""
    done = set()
    if not os.path.exists(out_jsonl):
        return done
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("__key__")
                if k:
                    done.add(k)
            except Exception:
                continue
    return done


# ===========================
# Normalization helpers
# ===========================

def canonical_relation(rel: str) -> str:
    """
    Map various string forms to our label inventory.
    """
    if not isinstance(rel, str):
        return ""
    r = rel.strip().lower()
    mapping = {
        "support": "Support",
        "supports": "Support",
        "attack": "Attack",
        "attacks": "Attack",
        "counter": "Attack",
        "counterargument": "Attack",
        "restatement": "Restatement",
        "rephrase": "Restatement",
        "paraphrase": "Restatement",
        "non-logical": "Non-logical",
        "nonlogical": "Non-logical",
        "non_logical": "Non-logical",
        "none": "No-Relation",
        "no-relation": "No-Relation",
        "no_relation": "No-Relation",
        "no relation": "No-Relation",
    }
    if r in mapping:
        return mapping[r]
    # fallback to LABEL_CANON if exists
    if r in LABEL_CANON:
        return LABEL_CANON[r]
    return ""


def _norm_idx(v, n):
    try:
        i = int(v)
        if 0 <= i < n:
            return i
    except Exception:
        pass
    return None


def _norm_label(v: str):
    r = canonical_relation(v)
    if r in LABELS:
        return r
    return None


def _norm_conf(v):
    try:
        x = float(v)
        if x < 0:
            return 0.0
        if x > 1:
            return 1.0
        return x
    except Exception:
        return 0.0


def _norm_evidence(ev):
    # Expect: {"type":"explicit|implicit","connectives":[...],"rationale":"..."}
    out = {"type": "implicit", "connectives": [], "rationale": ""}
    if not isinstance(ev, dict):
        return out
    t = ev.get("type", "implicit")
    if isinstance(t, str) and t.strip().lower() in ("explicit", "implicit"):
        out["type"] = t.strip().lower()
    conns = ev.get("connectives", [])
    if isinstance(conns, list):
        out["connectives"] = [str(x) for x in conns if isinstance(x, (str, int, float))]
    rat = ev.get("rationale", "")
    if isinstance(rat, str):
        out["rationale"] = rat.strip()
    return out


# ===========================
# Qwen Client with retry
# ===========================

class QwenClient:
    def __init__(self, base_url: str, api_key: str, timeout_sec: float = 180.0):
        if OpenAI is None:
            raise RuntimeError("Please `pip install openai==1.*` first.")
        if not api_key:
            raise RuntimeError("API key missing.")

        transport = httpx.HTTPTransport(retries=0, http2=False)
        http_client = httpx.Client(
            verify=certifi.where(),
            proxies=None,
            timeout=httpx.Timeout(timeout_sec, connect=30.0),
            transport=transport,
            trust_env=False
        )
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        self.timeout_sec = timeout_sec

    def _request_with_retry(self, *, model, messages, response_format, max_retries=6):
        delay = 1.0
        backoff = 1.6
        last_err = None
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    top_p=0.3,
                    response_format=response_format,
                    timeout=self.timeout_sec
                )
            except (APITimeoutError, APIConnectionError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_err = e
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay + random.random() * 0.5)
                delay *= backoff
            except RateLimitError as e:
                last_err = e
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay + random.random() * 0.7)
                delay *= backoff
            except APIStatusError as e:
                last_err = e
                if e.status_code >= 500 or e.status_code in (408, 409, 429, 499, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay + random.random() * 0.7)
                    delay *= backoff
                else:
                    raise
        raise last_err if last_err else RuntimeError("Unknown error in _request_with_retry")

    def call_ari(self, adus: List[str], model: str) -> Dict[str, Any]:
        N = len(adus)
        adus_block = "\n".join([f"[{i}] {adus[i]}" for i in range(N)])

        user = f"""
Task: Identify directed argumentative relations (ARI) among the given ADUs (0-based indices),
and classify each relation into one of the following labels:

  - Support
  - Attack
  - Restatement
  - Non-logical
  - No-Relation

ADU = minimal argumentative discourse unit (one proposition or meta-discourse segment).

OVERALL STRATEGY:
- Reconstruct the main reasoning chain: connect premises/reasons to intermediate conclusions
  and to final answers or option evaluations.
- Edges may connect NON-ADJACENT ADUs; do not restrict yourself to neighboring indices.
- Use a SMALL but MEANINGFUL set of edges: enough to capture the reasoning structure,
  but avoid fully connecting all ADUs.

LABEL SEMANTICS (VERY IMPORTANT):

1) Support
   - ADU A gives a reason, explanation, or piece of evidence for accepting ADU B.
   - Typical patterns: 'because', 'since', 'therefore', 'thus', 'so', 'implies that',
     'this means that', 'as a result', 'hence', 'therefore we know that ...'.
   - If A provides information that is clearly USED to justify B in the reasoning,
     annotate A → B as Support.

2) Attack (RARE, USE SPARINGLY)
   - ADU A DIRECTLY disputes, denies, or rejects the content, truth, or correctness of ADU B.
   - Typical patterns:
       * 'This is wrong / false / not true / not correct'
       * 'Option B is incorrect / should NOT be chosen'
       * 'It is NOT the case that ...' when B asserts the opposite.
   - A and B must be semantically INCOMPATIBLE (they cannot both be true).
   - DO NOT label as Attack when:
       * A merely presents a different possibility without explicitly refuting B;
       * A is just a new piece of reasoning leading to a different option;
       * A is a meta-comment (e.g., 'Some people might disagree').
   - If you are unsure between Attack vs Support/No-Relation, prefer Support or No-Relation
     rather than Attack.

3) Restatement
   - ADU A restates ADU B with the SAME meaning, without adding new conditions or changing scope.
   - No clear inferential marker (like 'therefore', 'implies that', 'this means that').
   - Examples: 'In other words, ...', 'That is to say, ...', or repeating the same
     logical content with slightly different wording.
   - Use Restatement only when the second ADU does NOT add extra logical constraints.

4) Non-logical (VERY RARE)
   - Use ONLY when there is a clear non-argumentative relation that is NOT used as a reason
     to accept/reject any claim:
       * Pure background or scene-setting that is not used in subsequent inference;
       * Pure meta-discourse about structuring the explanation (e.g. 'Now we analyze the options.'),
         when it is not itself a reason or claim about the logical content.
   - If an ADU gives a definition, example, or explanation that is actually USED to derive a
     conclusion (e.g., defining 'living gifts' and then using this definition to reason about
     the problem), you MUST treat the relation as Support instead of Non-logical.
   - If unsure whether a relation is Non-logical vs Support, prefer Support or No-Relation.
   - Non-logical edges should be RARE.

5) No-Relation
   - Default when there is no clear argumentative link between two ADUs.
   - It is better to output No-Relation than to hallucinate Attack or Non-logical.

INFERENTIAL CONNECTIVE RULE (CRUCIAL):
- If an ADU is introduced with 'implies that', 'this means that', 'therefore', 'thus',
  'hence', 'as a result', 'so (we can conclude that)', or similar:
  → You MUST annotate a relation **Support (source → target)** between the ADU(s) that
    provide the basis and the ADU that states the conclusion.
  → NEVER annotate such relations as 'Restatement'.
  → Even if the conclusion seems equivalent to previous content, treat it as Support
    (possibly a trivial inference).

EVALUATION & OPTIONS (MULTIPLE-CHOICE STYLE):
- If an ADU explicitly evaluates an option/statement (e.g., 'This option is correct/true',
  'Option B is wrong/false/invalid', 'Therefore, the answer must be D'):
  * The evaluation ADU should have at LEAST one incoming edge:
      - from ADU(s) that give the main reason why the option is correct/incorrect.
  * Positive evaluation → Support (reason ADU → evaluation ADU).
  * Negative evaluation → Attack (reason ADU → evaluation ADU).
- When an ADU describes the content of an option (e.g., 'Option A says that ...') and
  another ADU restates this content or evaluates it, connect them:
  * Pure repetition → Restatement.
  * Repetition + clear stance ('so this is true/false') → Support or Attack from the
    reasoning ADU to the option content ADU.

CHAIN CONSTRUCTION GUIDELINES:
- For each ADU that contains a clear conclusion marker ('therefore', 'thus', 'so', 'hence',
  'as a result', 'which means that', 'we can conclude that'):
  * Identify the earlier ADU(s) that provide the basis and connect them with Support edges.
- For each ADU that gives a key numerical or logical conclusion (e.g., 'Therefore, there are
  at least 13 female teachers', 'Thus, only option D satisfies all conditions'):
  * Connect from the main reasoning step that immediately precedes this conclusion.
- Meta-discourse ADUs such as 'Now, let's analyze each option:' or 'Next, we consider B'
  usually have NO argumentative edges; annotate them as No-Relation unless they clearly
  function as reasons or conclusions.

DIRECTIONALITY OF 'since/because/for':
- Clauses introduced by 'since', 'because', 'for' express reasons for the main claim.
- Direction: reason → claim (Support).
- Do NOT reverse the direction.

Rules for output graph:
1) Build a directed graph over ADUs. A relation is a tuple (src, tgt, label, confidence).
2) Edges can connect non-adjacent indices. Always prefer semantically motivated edges over adjacency.
3) If you CANNOT justify a relation by explicit or strong implicit evidence, choose 'No-Relation'.
4) Labels must be chosen from label_inventory (EXACT match).
5) confidence ∈ [0,1]. Use higher values only when connective evidence is clear/strong.
6) Indices must be 0..{N - 1}.

Schema (STRICT JSON):
{{
  "nodes": [
    {{"id": int, "text": "string"}}
  ],
  "edges": [
    {{
      "src": int,
      "tgt": int,
      "label": "Support|Attack|Restatement|Non-logical|No-Relation",
      "confidence": float,
      "evidence": {{
        "type": "explicit|implicit",
        "connectives": ["string", "..."],
        "rationale": "short justification referencing the ADU wording"
      }}
    }}
  ]
}}

label_inventory with definitions:
{json.dumps(LABELS, ensure_ascii=False)}

ADUs (N={N}):
{adus_block}

Return STRICT JSON only.
""".strip()

        messages = [
            {"role": "system", "content": "You are an expert in argument mining. Output strict JSON only."},
            {"role": "user", "content": user}
        ]
        resp = self._request_with_retry(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content
        try:
            data = json.loads(txt)
        except Exception:
            cleaned = txt.strip().strip('`').replace("\n", " ").replace("\r", " ")
            s, e = cleaned.find("{"), cleaned.rfind("}")
            if s >= 0 and e > s:
                data = json.loads(cleaned[s:e + 1])
            else:
                raise ValueError("Model did not return valid JSON.")
        return data


# ===========================
# Evaluation
# ===========================

def build_gold_edge_set(item: Dict[str, Any], n_adus: int):
    """
    从 DATA.json 的 item 里抽取 gold edges:
      item["edges"] = [{"source": int, "target": int, "relation": str}, ...]
    返回:
      gold_edges_labeled: set[(src, tgt, label)]
      dict_gold_label_by_pair: {(src,tgt): label}
    """
    edges_raw = item.get("edges", [])
    gold_edges = set()
    label_by_pair = {}

    for e in edges_raw:
        if not isinstance(e, dict):
            continue
        src = _norm_idx(e.get("source"), n_adus)
        tgt = _norm_idx(e.get("target"), n_adus)
        if src is None or tgt is None:
            continue
        rel = _norm_label(e.get("relation"))
        if not rel:
            continue
        # 一般 gold 中不会有 No-Relation；如果有就忽略
        if rel == "No-Relation":
            continue
        gold_edges.add((src, tgt, rel))
        label_by_pair[(src, tgt)] = rel
    return gold_edges, label_by_pair


def build_pred_edge_set(data: Dict[str, Any], n_adus: int):
    """
    从模型输出 data 里抽取规范化的预测边:
    只返回非 No-Relation 边。
    """
    edges_in = data.get("edges", [])
    if not isinstance(edges_in, list):
        edges_in = []

    pred_edges = set()
    label_by_pair = {}

    for e in edges_in:
        if not isinstance(e, dict):
            continue
        src = _norm_idx(e.get("src"), n_adus)
        tgt = _norm_idx(e.get("tgt"), n_adus)
        if src is None or tgt is None:
            continue
        lbl = _norm_label(e.get("label"))
        if not lbl:
            continue
        if lbl == "No-Relation":
            # No-Relation 不加入正类集合
            continue
        pred_edges.add((src, tgt, lbl))
        label_by_pair[(src, tgt)] = lbl

    return pred_edges, label_by_pair


def compute_micro_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


# ===========================
# Main pipeline
# ===========================

def run_pipeline(input_path: str,
                 model: str,
                 base_url: str,
                 api_key: str,
                 out_dir: str,
                 max_items: int = 0):
    os.makedirs(out_dir, exist_ok=True)

    # 固定 JSONL 名称用于断点续跑
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = os.path.join(out_dir, f"ari_eval.jsonl")

    # 带时间戳的 log 文件
    log_path = os.path.join(out_dir, f"aric_eval_log_{ts}.txt")

    done_keys = load_done_keys(out_jsonl)

    qwen = QwenClient(base_url=base_url, api_key=api_key)

    fout = open(out_jsonl, "a", encoding="utf-8")
    flog = open(log_path, "w", encoding="utf-8")

    # 写 log 头部信息
    header = (
        f"[ARIC-EVAL LOG]\n"
        f"timestamp: {ts}\n"
        f"input: {input_path}\n"
        f"model: {model}\n"
        f"base_url: {base_url}\n"
        f"output_jsonl: {out_jsonl}\n"
        f"out_dir: {out_dir}\n"
        f"max_items: {max_items}\n"
        f"label_inventory: {list(LABELS.keys())}\n"
        f"{'-'*70}\n"
    )
    print(header, end="")
    flog.write(header)

    # -------- 全局统计量 --------
    # ARIC（有标签边）的 TP/FP/FN
    global_tp_labeled = 0
    global_fp_labeled = 0
    global_fn_labeled = 0

    # ARI（unlabeled）统计
    global_tp_ari = 0
    global_fp_ari = 0
    global_fn_ari = 0

    # ARC 统计（仅在端点正确的前提下）
    global_arc_correct = 0
    global_arc_wrong = 0
    per_label_arc_correct = Counter()
    per_label_arc_wrong_gold = Counter()

    # label-wise confusion for ARIC
    label_tp = Counter()
    label_fp = Counter()
    label_fn = Counter()

    processed = 0

    for seq, item in enumerate(load_items(input_path)):
        key = make_item_key(item, seq)
        if key in done_keys:
            continue
        if max_items > 0 and processed >= max_items:
            break

        adus = item.get("adus_text") or []
        if not isinstance(adus, list) or not adus:
            msg = f"[WARN] item {key} has no adus_text, skip.\n"
            print(msg, end="")
            flog.write(msg)
            continue
        adus = [str(x) for x in adus]

        n_adus = len(adus)
        nodes = [{"id": i, "text": t} for i, t in enumerate(adus)]

        gold_edges_labeled, gold_label_by_pair = build_gold_edge_set(item, n_adus)
        if not gold_edges_labeled:
            msg = f"[WARN] item {key} has no gold edges, skip.\n"
            print(msg, end="")
            flog.write(msg)
            continue

        gold_unlab = {(s, t) for (s, t, l) in gold_edges_labeled}

        # Call LLM
        try:
            data = qwen.call_ari(adus, model=model)
        except Exception as e:
            result = {
                "__key__": key,
                "id": item.get("id"),
                "idx": item.get("idx"),
                "nodes": nodes,
                "gold_edges": sorted(list(gold_edges_labeled)),
                "pred_edges": [],
                "error": f"{type(e).__name__}: {str(e)[:300]}"
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            msg = f"[ERROR] item {key} LLM error: {type(e).__name__}: {str(e)[:200]}\n"
            print(msg, end="")
            flog.write(msg)

            processed += 1
            continue

        pred_edges_labeled, pred_label_by_pair = build_pred_edge_set(data, n_adus)
        pred_unlab = {(s, t) for (s, t, l) in pred_edges_labeled}

        # ---------- ARI（unlabeled edges） ----------
        tp_ari_case = len(gold_unlab & pred_unlab)
        fp_ari_case = len(pred_unlab - gold_unlab)
        fn_ari_case = len(gold_unlab - pred_unlab)

        global_tp_ari += tp_ari_case
        global_fp_ari += fp_ari_case
        global_fn_ari += fn_ari_case

        ari_case_metrics = compute_micro_metrics(tp_ari_case, fp_ari_case, fn_ari_case)

        # ---------- ARIC（有标签边） ----------
        tp_edges = gold_edges_labeled & pred_edges_labeled
        fp_edges = pred_edges_labeled - gold_edges_labeled
        fn_edges = gold_edges_labeled - pred_edges_labeled

        tp_l = len(tp_edges)
        fp_l = len(fp_edges)
        fn_l = len(fn_edges)

        global_tp_labeled += tp_l
        global_fp_labeled += fp_l
        global_fn_labeled += fn_l

        # label-wise confusion for ARIC
        for (s, t, lbl) in tp_edges:
            label_tp[lbl] += 1
        for (s, t, lbl) in fp_edges:
            label_fp[lbl] += 1
        for (s, t, lbl) in fn_edges:
            label_fn[lbl] += 1

        aric_case_metrics = compute_micro_metrics(tp_l, fp_l, fn_l)

        # ---------- ARC（label classification 给边打标签） ----------
        # 只看端点 (s,t) 正确的交集
        correct_pairs = gold_unlab & pred_unlab
        arc_correct_case = 0
        arc_wrong_case = 0

        for (s, t) in correct_pairs:
            g_lbl = gold_label_by_pair.get((s, t))
            p_lbl = pred_label_by_pair.get((s, t))
            if g_lbl is None or p_lbl is None:
                continue
            if g_lbl == p_lbl:
                arc_correct_case += 1
                per_label_arc_correct[g_lbl] += 1
            else:
                arc_wrong_case += 1
                per_label_arc_wrong_gold[g_lbl] += 1

        global_arc_correct += arc_correct_case
        global_arc_wrong += arc_wrong_case

        arc_total_case = arc_correct_case + arc_wrong_case
        arc_acc_case = arc_correct_case / arc_total_case if arc_total_case > 0 else 0.0

        # ---------- 打印 + 写 log ----------
        msg_case = (
            f"[CASE {key}] "
            f"GoldEdges={len(gold_edges_labeled)}, PredEdges={len(pred_edges_labeled)} | "
            f"ARI: TP={tp_ari_case}, FP={fp_ari_case}, FN={fn_ari_case}, "
            f"P={ari_case_metrics['precision']:.3f}, R={ari_case_metrics['recall']:.3f}, F1={ari_case_metrics['f1']:.3f} | "
            f"ARIC(labeled): TP={tp_l}, FP={fp_l}, FN={fn_l}, "
            f"P={aric_case_metrics['precision']:.3f}, R={aric_case_metrics['recall']:.3f}, F1={aric_case_metrics['f1']:.3f} | "
            f"ARC(acc on detected)={arc_acc_case:.3f} (correct={arc_correct_case}, wrong={arc_wrong_case})\n"
        )
        print(msg_case, end="")
        flog.write(msg_case)

        result = {
            "__key__": key,
            "id": item.get("id"),
            "idx": item.get("idx"),
            "nodes": nodes,
            "gold_edges": sorted(list(gold_edges_labeled)),
            "pred_edges": sorted(list(pred_edges_labeled)),
            "metrics": {
                "ari_case": {
                    "tp": tp_ari_case,
                    "fp": fp_ari_case,
                    "fn": tp_ari_case and tp_ari_case * 0 + fn_ari_case,  # keep explicit
                    **ari_case_metrics
                },
                "aric_case": {
                    "tp": tp_l,
                    "fp": fp_l,
                    "fn": fn_l,
                    **aric_case_metrics
                },
                "arc_case": {
                    "correct": arc_correct_case,
                    "wrong": arc_wrong_case,
                    "accuracy": arc_acc_case
                }
            }
        }
        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        fout.flush()

        processed += 1

    fout.close()

    # ---------- Overall metrics ----------
    # ARI
    overall_ari = compute_micro_metrics(global_tp_ari, global_fp_ari, global_fn_ari)
    overall_ari.update({"tp": global_tp_ari, "fp": global_fp_ari, "fn": global_fn_ari})

    # ARIC (labeled edges)
    overall_aric = compute_micro_metrics(global_tp_labeled, global_fp_labeled, global_fn_labeled)
    overall_aric.update({
        "tp": global_tp_labeled,
        "fp": global_fp_labeled,
        "fn": global_fn_labeled
    })

    # ARC
    arc_total = global_arc_correct + global_arc_wrong
    overall_arc_acc = global_arc_correct / arc_total if arc_total > 0 else 0.0

    per_label_arc_metrics = {}
    for lbl in ["Support", "Attack", "Restatement", "Non-logical"]:
        c = per_label_arc_correct[lbl]
        w = per_label_arc_wrong_gold[lbl]
        total_lbl = c + w
        acc_lbl = c / total_lbl if total_lbl > 0 else 0.0
        per_label_arc_metrics[lbl] = {
            "correct": c,
            "wrong": w,
            "total_detected": total_lbl,
            "accuracy": acc_lbl
        }

    metrics_out = {
        "processed_items": processed,
        "ARI_unlabeled": overall_ari,
        "ARIC_labeled_edges": overall_aric,
        "ARC_label_on_detected_edges": {
            "correct": global_arc_correct,
            "wrong": global_arc_wrong,
            "accuracy": overall_arc_acc,
            "per_label": per_label_arc_metrics
        },
        "ARIC_label_confusion": {
            "label_tp": dict(label_tp),
            "label_fp": dict(label_fp),
            "label_fn": dict(label_fn),
        }
    }

    metrics_json_ts = os.path.join(out_dir, f"ari_eval_overall_{ts}.json")
    with open(metrics_json_ts, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    # 打印 + 写入 log 尾部
    tail = []
    tail.append("\n========== Overall ARI / ARIC / ARC Metrics ==========\n")
    tail.append(f"Processed items: {processed}\n")
    tail.append("\n[ARI - unlabeled edges]\n")
    tail.append(f"  TP={overall_ari['tp']}, FP={overall_ari['fp']}, FN={overall_ari['fn']}\n")
    tail.append(f"  Precision={overall_ari['precision']:.4f}, Recall={overall_ari['recall']:.4f}, F1={overall_ari['f1']:.4f}\n")

    tail.append("\n[ARIC - labeled edges (位置+标签同时正确)]\n")
    tail.append(f"  TP={overall_aric['tp']}, FP={overall_aric['fp']}, FN={overall_aric['fn']}\n")
    tail.append(f"  Precision={overall_aric['precision']:.4f}, Recall={overall_aric['recall']:.4f}, F1={overall_aric['f1']:.4f}\n")

    tail.append("\n[ARC - label classification on correctly detected edges]\n")
    tail.append(f"  correct={global_arc_correct}, wrong={global_arc_wrong}, "
                f"accuracy={overall_arc_acc:.4f}\n")
    tail.append("  per-label ARC accuracy (only on edges whose endpoints are correctly detected):\n")
    for lbl, m in per_label_arc_metrics.items():
        tail.append(
            f"    [{lbl}] correct={m['correct']}, wrong={m['wrong']}, "
            f"total={m['total_detected']}, acc={m['accuracy']:.4f}\n"
        )
    tail.append("======================================================\n")

    tail_str = "".join(tail)
    print(tail_str, end="")
    flog.write(tail_str)
    flog.close()

    print(f"[INFO] Overall metrics saved to: {metrics_json_ts}")
    print(f"[INFO] Log saved to: {log_path}")


def main():
    ap = argparse.ArgumentParser(description="ARIC evaluation from DATA.json with Qwen")
    ap.add_argument("--input", default="DATA.json",
                    help="Path to DATA.json (or JSONL) with adus_text and gold edges")
    ap.add_argument("--model", default="qwen3-max", help="Qwen model name")
    ap.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    help="DashScope OpenAI-compatible URL")
    ap.add_argument("--api-key", default="", help="API key (or use env DASHSCOPE_API_KEY)")
    ap.add_argument("--out", default="results_aric", help="Output directory")
    ap.add_argument("--max-items", type=int, default=0,
                    help="Max items to evaluate (0 = all)")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing API key. Use --api-key or set DASHSCOPE_API_KEY.")

    run_pipeline(
        input_path=args.input,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        out_dir=args.out,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
