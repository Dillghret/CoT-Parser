import os, re, csv, json, time, random, argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from datetime import datetime  # 用于时间戳

# ================= Config =================
MODEL_NAME = "qwen3-max"
BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEMPERATURE = 0.7
MAX_RETRIES = 2
REQUEST_SLEEP_SEC = 0.20

K_EXPLICIT = 1
K_IMPLICIT = 1
K_NOSPLIT  = 1

TOKEN_MIN_WORDS = 3  # minimal tokens per ADU after split

WORD = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?"


# ================= Utilities =================
def strip_angle_tags(s: str) -> str:
    return re.sub(r"<[^>]*>", "", s or "")


def normalize_spaces(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    return s.strip()


def preprocess(raw: str) -> str:
    return normalize_spaces(strip_angle_tags(raw))


def count_tokens_simple(s: str) -> int:
    return len(re.findall(WORD, s))


# ================= Sentence split (English) =================
def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    """
    Return [(start,end,sentence_text)] using English punctuation .?!;:
    Fallback to chunking by length if the text has almost no sentence punctuation.
    """
    t = text
    parts = re.split(r'(?<=[\.\?\!;:])\s+', t)
    parts = [p.strip() for p in parts if p and p.strip()]
    spans = []

    if len(parts) <= 1:
        CHUNK = 160
        i = 0
        while i < len(t):
            j = min(len(t), i + CHUNK)
            if j < len(t):
                k = t.find(" ", j)
                if k != -1 and k - i <= 220:
                    j = k
            seg = t[i:j].strip()
            if seg:
                s = t.find(seg, i, j)
                if s == -1:
                    s = i
                spans.append((s, s + len(seg), seg))
            i = j + 1
        return spans

    search_from = 0
    for piece in parts:
        pos = t.find(piece, search_from)
        if pos == -1:
            pos = search_from
        end = pos + len(piece)
        spans.append((pos, end, piece))
        search_from = end
    return spans


# ================= Text alignment helpers =================
def expand_to_word_boundaries(sentence: str, s: int, e: int) -> Tuple[int, int]:
    # left expand if cutting inside a word
    while s > 0 and sentence[s - 1].isalnum() and sentence[s].isalnum():
        s -= 1
    # right expand
    n = len(sentence)
    while e < n and e > 0 and sentence[e - 1].isalnum() and sentence[e].isalnum():
        e += 1
    # trim spaces
    while s < e and sentence[s].isspace():
        s += 1
    while e > s and sentence[e - 1].isspace():
        e -= 1
    return s, e


def find_spans_by_texts(sentence: str, adu_texts: List[str]) -> List[Tuple[int, int]]:
    """
    Align adu_texts back to sentence. Greedy left-to-right, case-insensitive,
    whitespace-flexible. Avoid overlaps by advancing search window.
    """
    spans: List[Tuple[int, int]] = []
    last_end = 0
    for t in adu_texts:
        t_clean = (t or "").strip()
        if not t_clean:
            continue

        pat_exact = re.escape(t_clean)
        pat_ws    = re.escape(t_clean).replace(r"\ ", r"\s+")

        m = re.search(pat_exact, sentence[last_end:])
        base = last_end
        if not m:
            m = re.search(pat_ws, sentence[last_end:], flags=re.I)
        if not m:
            m = re.search(pat_ws, sentence, flags=re.I)
            base = 0
        if not m:
            continue

        s = base + m.start()
        e = base + m.end()
        s, e = expand_to_word_boundaries(sentence, s, e)

        if spans and s < spans[-1][1]:
            s = spans[-1][1]
            if s >= e:
                continue
            s, e = expand_to_word_boundaries(sentence, s, e)

        spans.append((s, e))
        last_end = e

    if not spans:
        spans = [(0, len(sentence))]
    return spans


# ================= PDTB sampling for few-shot =================
@dataclass
class FewShot:
    sentence: str
    adus_text: List[str]


FIXED_FEW_SHOTS: List[FewShot] = [
    FewShot(
        sentence="Big indexer Bankers Trust Co. also uses futures in a strategy that on average has added one percentage point to its enhanced fund's returns.J. Thomas Allen, president of Pittsburgh-based Advanced Investment Management Inc., agrees it's a good idea to jump between the S&P 500 stocks and futures But some indexers make little or no use of futures, saying that these instruments present added risks for investors",
        adus_text=[
            "Big indexer Bankers Trust Co. also uses futures in a strategy that on average has added one percentage point to its enhanced fund's returns.J. Thomas Allen, president of Pittsburgh-based Advanced Investment Management Inc., agrees it's a good idea to jump between the S&P 500 stocks and futures",
            "some indexers make little",
            "no use of futures, saying that these instruments present added risks for investors",
        ],
    ),
    FewShot(
        sentence="Don't take this as some big opening for major movement on economic cooperation, or arms control, or the environment Those things will all come up, but in a fairly informal way",
        adus_text=[
            "Don't take this as some big opening for major movement on economic cooperation, or arms control, or the environment",
            "Those things will all come up, but in a fairly informal way",
        ],
    ),
    FewShot(
        sentence="Mr. Kennedy failed to get his amendment incorporated into last year's anti-drug legislation, and it will be severely attacked on the Senate floor this time around",
        adus_text=[
            "Mr. Kennedy failed to get his amendment incorporated into last year's anti-drug legislation, and it will be severely attacked on the Senate floor this time around",
        ],
    ),
]


def build_sentence_from_pdtb(arg1: str, conn: str, arg2: str) -> Optional[FewShot]:
    arg1 = (arg1 or "").strip()
    arg2 = (arg2 or "").strip()
    conn = (conn or "").strip()
    if not arg1 or not arg2:
        return None

    if conn:
        s = f"{arg1} {conn} {arg2}"

        def split_inside(x: str) -> List[str]:
            parts = re.split(r"\s+(and|or|but|then)\s+", x, flags=re.I)
            out, buf, i = [], "", 0
            while i < len(parts):
                if i == 0:
                    buf = parts[i]
                elif parts[i].lower() in ("and", "or", "but", "then"):
                    if buf.strip():
                        out.append(buf.strip())
                    buf = parts[i + 1] if i + 1 < len(parts) else ""
                    i += 1
                i += 1
            if buf.strip():
                out.append(buf.strip())
            out = [z for z in out if count_tokens_simple(z) >= 3]
            return out or [x]

        adus = split_inside(arg1) + split_inside(arg2)
    else:
        s = f"{arg1} {arg2}"
        adus = [arg1, arg2]

    s = preprocess(s)
    return FewShot(s, adus)


def sample_fewshots_from_pdtb(csv_path: str, k_exp: int, k_imp: int, k_nosplit: int) -> List[FewShot]:
    required_cols = {"Arg1_RawText", "Arg2_RawText", "Connective_RawText"}
    exp, imp = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"PDTB csv requires columns {sorted(required_cols)}, got {reader.fieldnames}")
        for row in reader:
            a1 = row.get("Arg1_RawText") or ""
            a2 = row.get("Arg2_RawText") or ""
            conn = row.get("Connective_RawText") or ""
            fs = build_sentence_from_pdtb(a1, conn, a2)
            if not fs:
                continue
            if conn.strip():
                exp.append(fs)
            else:
                imp.append(fs)

    random.shuffle(exp)
    random.shuffle(imp)
    shots = exp[:k_exp] + imp[:k_imp]

    # no-split: Arg1/Arg2 单句
    nos = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ("Arg1_RawText", "Arg2_RawText"):
                t = preprocess(row.get(col) or "")
                if count_tokens_simple(t) >= 6:
                    nos.append(FewShot(t, [t]))
    random.shuffle(nos)
    shots += nos[:k_nosplit]

    if not shots:
        shots = [FewShot("If demand falls then profits drop.",
                         ["If demand falls", "profits drop"])]
    return shots


def render_fewshot_block(shots: List[FewShot]) -> str:
    """
    把 few-shots 渲染成文本，拼在 prompt 里。
    这里使用固定 few-shots，避免随机采样。
    """
    bad_good_block = (
        "DON'T vs DO (contrastive):\n"
        "- BAD: \", bu\"  →  GOOD: \"but ...\" (never cut inside a connective)\n"
        "- BAD: \", nor does i\"  →  GOOD: \", nor does it ...\"\n"
        "- BAD: \", which implies that\"  →  GOOD: \"which implies that ...\" as PART of a full clause\n"
        "- BAD: \"A.\" alone  →  GOOD: \"A. full option sentence\" as one ADU\n"
        "- BAD: \"Scientists:\" alone  →  GOOD: \"Scientists: ...\" as one ADU\n"
        "- BAD: splitting a definition like \"X refers to A or B\" into two ADUs\n"
        "       →  GOOD: keep the whole definition as ONE ADU if it defines a single concept."
    )

    parts = [bad_good_block, "Examples:"]
    for fs in shots:
        demo = {"adus_text": fs.adus_text}
        parts.append(
            "Sentence:\n" + fs.sentence + "\n" +
            "Expected:\n" + json.dumps(demo, ensure_ascii=False)
        )
    return "\n\n".join(parts)


# ================= LLM call & cleaning =================
INSTRUCTION = (
    "Task: For ONE English sentence from a logical reasoning solution, split it into minimal, "
    "self-contained argument spans (ADUs).\n"
    "\n"
    "General goal:\n"
    "- Each ADU is a single, complete proposition that could later be labeled as a premise or a claim.\n"
    "- Keep the segmentation at proposition level, BUT avoid over-fragmentation and extremely small spans.\n"
    "- When in doubt, prefer slightly FEWER ADUs rather than too many tiny fragments.\n"
    "\n"
    "STRICT RULES (must all hold):\n"
    "1) Output ONLY valid JSON: {\"adus_text\": [\"...\", \"...\", ...]} with 1 or more spans.\n"
    "2) Each span MUST be a VERBATIM substring of the given sentence (no rewriting, no reordering), "
    "   trimmed of leading/trailing spaces.\n"
    "3) NEVER cut inside a token/word (no partial words like \"conc\", \"inherite\").\n"
    "4) NEVER output punctuation-only or conjunction-only fragments (e.g. ',', 'and', 'or', 'but').\n"
    "5) Each ADU should correspond to ONE main proposition that can be judged true/false. Do NOT split into\n"
    "   multiple ADUs if the parts cannot be meaningfully judged independently.\n"
    "\n"
    "Connectives and clauses:\n"
    "6) Clausal connectives such as 'and', 'or', 'but', 'if', 'since', 'because', 'although', 'when', 'while', "
    "   'therefore', 'thus', 'hence', 'however', 'despite', 'as a result', 'this means that', 'which implies that', "
    "   etc. OFTEN indicate a boundary between two propositions ONLY IF BOTH SIDES are full clauses that can stand "
    "   alone as propositions.\n"
    "   - Do NOT split inside short lists or within a single complex definition (e.g. "
    "     'economic cooperation, or arms control, or the environment' should stay in ONE ADU).\n"
    "7) Subordinate / relative clauses introduced by 'which', 'that', 'who', 'because', 'if', 'then', etc. MUST be "
    "   COMPLETE clauses if you return them as ADUs. Do NOT output only the introducer or half a clause.\n"
    "8) Inferential markers like 'this implies that', 'which means that', 'which implies that', 'as a result', "
    "   'therefore', 'thus', 'hence':\n"
    "   - You MAY start a new ADU at the main clause AFTER the marker,\n"
    "   - BUT you MUST NOT output the marker alone as an ADU,\n"
    "   - AND you MUST NOT cut between the marker and its following clause.\n"
    "\n"
    "Discourse connectives and meta phrases:\n"
    "9) Discourse connectives at the beginning of a span (such as 'However', 'Therefore', 'Thus', "
    "   'In conclusion', 'Overall') MAY be excluded from the ADU text. It is acceptable to start the ADU at the "
    "   first content word after such a connective.\n"
    "   - Phrases like 'In option A,' or 'Option B' or 'Scientists:' or 'Critic:' are NOT standalone ADUs. "
    "     They MUST stay attached to the main clause they introduce in the SAME ADU.\n"
    "10) After a comma, pronouns like 'we', 'I', 'they', 'it', 'this', 'that' belong to the FOLLOWING ADU "
    "    (e.g. '..., we conclude that ...' → the ADU starts from 'we conclude that ...').\n"
    "11) Meta-discourse such as 'Now, let's analyze each option:' or 'Next, we consider ...' may be kept as its own "
    "    ADU IF it appears as a complete clause. Do NOT produce small fragments like 'Now,' or 'In option A,' alone.\n"
    "\n"
    "Evaluation, reasons, and restatements:\n"
    "12) Evaluation + reason structures can be split into two ADUs when both sides are full propositions: "
    "    one ADU for the evaluation/claim (e.g. 'This option is correct.'), and another ADU for the reason/explanation "
    "    (e.g. 'because ...', 'since ...'). Do NOT split if the reason clause is incomplete on its own.\n"
    "13) ADUs that are pure restatements or paraphrases of a previous proposition but introduced with inferential "
    "    markers ('implies that', 'this means that', 'therefore', 'thus', 'hence') MAY be kept as a separate ADU to "
    "    represent the inferential step, provided the resulting span is a full proposition.\n"
    "\n"
    "Special rules for multiple-choice reasoning text:\n"
    "14) Option labels such as 'A.', 'B.', 'C.', 'D.', '(A)', '(B)', or phrases like 'In option A,' / 'Option B' "
    "    MUST NOT be separated from the statement they introduce. The label + the full option sentence SHOULD be "
    "    returned as ONE ADU.\n"
    "    - BAD: 'A.' as an ADU and 'Some young teachers are not women' as another ADU.\n"
    "    - GOOD: 'A. Some young teachers are not women' as one ADU.\n"
    "15) For lines like 'answer: D' or 'Answer: C', the entire phrase should be treated as at most ONE ADU. "
    "    Do NOT split at the colon or between 'answer' and the letter.\n"
    "16) Speaker labels like 'Scientists:' or 'Critic:' MUST stay attached to the clause they introduce. "
    "    Never output 'Scientists:' alone as an ADU.\n"
    "17) Definition-like sentences (e.g. 'Living gifts refer to the donation of money to children, grandchildren, "
    "    and other relatives when the donor decides to live, or to use it for vacations and the establishment of "
    "    trust funds.') SHOULD normally be kept as ONE ADU if the parts jointly define the same concept and are not "
    "    intended as separate independent claims.\n"
    "\n"
    "Length and completeness:\n"
    "18) Each ADU should contain at least 3 meaningful tokens (content words) and form a coherent span that can be "
    "    judged true/false as a proposition.\n"
    "19) Do NOT delete any content from the sentence; you only choose spans. Every returned span must appear exactly "
    "    somewhere in the input sentence.\n"
    "20) When ambiguous, lean towards FEWER, larger ADUs rather than too many tiny fragments, as long as each ADU is "
    "    still a single proposition.\n"
    "\n"
    "Output only the JSON object."
)


def make_client() -> OpenAI:
    api_key = ""
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def call_llm_sentence(client: OpenAI, sentence: str, fewshot_block: str) -> List[str]:
    user_prompt = (
        INSTRUCTION + "\n\n"
        "Few-shot examples:\n" + fewshot_block + "\n\n"
        "Now segment this sentence and output ONLY JSON with key 'adus_text':\n" + sentence
    )
    last_err = None
    for _ in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a precise clause splitter. Only output valid JSON."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
            )
            out = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", out, flags=re.S)
            if not m:
                raise ValueError("no JSON found")
            data = json.loads(m.group(0))
            texts = data.get("adus_text", [])
            if not isinstance(texts, list):
                raise ValueError("bad 'adus_text'")
            clean = [str(x).strip() for x in texts if isinstance(x, str) and x.strip()]
            return clean or []
        except Exception as e:
            last_err = e
            time.sleep(0.3)
            continue
    print("[WARN] call_llm_sentence failed after retries:", last_err)
    return []


def looks_truncated(token: str) -> bool:
    return len(token) <= 2 or not re.fullmatch(WORD, token)


def is_punct_only_text(s: str) -> bool:
    return bool(re.fullmatch(r"\s*[\W_]+\s*", s or ""))


def clean_llm_adus(sentence: str, adus_text: List[str]) -> List[str]:
    out = []
    for t in adus_text:
        if not t or is_punct_only_text(t):
            continue
        t = t.strip()

        if t not in sentence:
            pat_ws = re.escape(t).replace(r"\ ", r"\s+")
            if not re.search(pat_ws, sentence, flags=re.I):
                continue

        start = sentence.find(t)
        if start < 0:
            m = re.search(re.escape(t).replace(r"\ ", r"\s+"), sentence, flags=re.I)
            if not m:
                continue
            start, end = m.start(), m.end()
        else:
            end = start + len(t)

        left_ok  = (start == 0) or (not sentence[start - 1].isalnum())
        right_ok = (end == len(sentence)) or (not sentence[end:end + 1].isalnum())
        tail = re.findall(WORD, t)
        tail_bad = (len(tail) == 0) or looks_truncated(tail[-1])
        starts_bad = bool(re.match(
            r"^(,|\band\b|\bor\b|\bbut\b|\bnor\b|\bas\b|\bwhich\b|\bthat\b|\bthen\b)\b",
            t.strip(), flags=re.I
        ))
        ends_bad = bool(re.search(
            r"(,|\band\b|\bor\b|\bbut\b|\bnor\b|\bas\b|\bwhich\b|\bthat\b|\bthen\b)\s*$",
            t.strip(), flags=re.I
        ))

        if left_ok and right_ok and not tail_bad and not starts_bad and not ends_bad:
            out.append(t)
        else:
            out.append(t)
    return out


# ================= Post-process one sentence =================
def merge_small_or_dependency_fragments(
    spans: List[Tuple[int, int]], sentence: str, min_tokens=3
) -> List[Tuple[int, int]]:
    if not spans:
        return spans
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = []
    i = 0
    while i < len(spans):
        s, e = spans[i]
        text = sentence[s:e]
        tok = re.findall(WORD, text)
        tail_bad = len(tok) == 0 or looks_truncated(tok[-1])
        if (len(tok) < min_tokens or tail_bad) and i + 1 < len(spans):
            s2, e2 = spans[i + 1]
            gap = sentence[e:s2]
            if not re.search(r"[A-Za-z0-9]", gap):
                s_new, e_new = expand_to_word_boundaries(sentence, s, e2)
                merged.append((s_new, e_new))
                i += 2
                continue
        merged.append((s, e))
        i += 1
    return merged


def postprocess_sentence(sentence: str, adu_texts: List[str]) -> List[Tuple[int, int, str]]:
    adu_texts = clean_llm_adus(sentence, adu_texts)
    spans = find_spans_by_texts(sentence, adu_texts)

    tmp = []
    for s, e in spans:
        if re.fullmatch(r"\s*[\W_]+\s*", sentence[s:e]):
            continue
        s, e = expand_to_word_boundaries(sentence, s, e)
        tmp.append((s, e))
    spans = tmp

    spans = merge_small_or_dependency_fragments(spans, sentence, min_tokens=TOKEN_MIN_WORDS)

    final = []
    for s, e in spans:
        if s >= e:
            continue
        if re.fullmatch(r"\s*[\W_]+\s*", sentence[s:e]):
            continue
        if count_tokens_simple(sentence[s:e]) < TOKEN_MIN_WORDS:
            continue
        final.append((s, e))

    if not final:
        final = [(0, len(sentence))]

    uniq = []
    for s, e in sorted(final, key=lambda x: (x[0], x[1])):
        if not uniq or (s, e) != uniq[-1]:
            uniq.append((s, e))
    return [(s, e, sentence[s:e]) for (s, e) in uniq]


# ================= Segment whole preprocessed text =================
def segment_preprocessed_text(client: OpenAI, text: str, fewshot_block: str) -> Dict[str, Any]:
    """
    对已经 preprocess 过的整段文本 text 做 ACs 切分。
    """
    sent_spans = split_sentences(text)
    sentences_out = []
    n_calls = 0

    for i, (s, e, sent) in enumerate(sent_spans, start=1):
        adu_texts = call_llm_sentence(client, sent, fewshot_block)
        n_calls += 1
        adus_local = postprocess_sentence(sent, adu_texts)  # [(a,b,t)]

        adus_global = []
        for (a, b, t) in adus_local:
            adus_global.append({
                "span": [s + a, s + b],
                "text": t
            })

        sentences_out.append({
            "sent_id": i,
            "span": [s, e],
            "text": sent,
            "adus": adus_global
        })
        time.sleep(REQUEST_SLEEP_SEC)

    return {"sentences": sentences_out, "n_api_calls": n_calls}


def segment_one_cot(client: OpenAI, raw_text: str, fewshot_block: str) -> Dict[str, Any]:
    """
    兼容旧接口：输入 raw_text（比如 CoT），内部 preprocess 再切分。
    """
    text = preprocess(raw_text)
    return segment_preprocessed_text(client, text, fewshot_block)


# ================= IO =================
def load_items(path: str) -> List[Dict[str, Any]]:
    """
    既支持 JSON list，也支持 JSONL。
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        if head.strip().startswith("["):
            arr = json.load(f)
            assert isinstance(arr, list)
            return arr
        else:
            return [json.loads(line) for line in f if line.strip()]


# ================= Evaluation helpers =================
def extract_pred_spans(seg_output: Dict[str, Any]) -> List[Tuple[int, int]]:
    spans = []
    for sent in seg_output.get("sentences", []):
        for adu in sent.get("adus", []):
            span = adu.get("span")
            if isinstance(span, list) and len(span) == 2:
                s, e = int(span[0]), int(span[1])
                if s < e:
                    spans.append((s, e))
    spans = sorted(set(spans), key=lambda x: (x[0], x[1]))
    return spans


def align_gold_spans(full_text: str, gold_adus: List[str]) -> List[Tuple[int, int]]:
    gold_clean = [preprocess(t) for t in gold_adus]
    spans = find_spans_by_texts(full_text, gold_clean)
    return spans


def spans_to_boundaries(spans: List[Tuple[int, int]], text_len: int) -> set:
    bounds = set()
    for s, e in spans:
        if 0 < e < text_len:
            bounds.add(e)
    return bounds


def compute_boundary_metrics(
    text_len: int,
    gold_spans: List[Tuple[int, int]],
    pred_spans: List[Tuple[int, int]],
) -> Dict[str, float]:
    gold_b = spans_to_boundaries(gold_spans, text_len)
    pred_b = spans_to_boundaries(pred_spans, text_len)

    tp = len(gold_b & pred_b)
    fp = len(pred_b - gold_b)
    fn = len(gold_b - pred_b)

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "gold_boundaries": len(gold_b),
        "pred_boundaries": len(pred_b),
    }


# ================= Main: evaluate ACs on DATA.json =================
def main():
    """
    评估 ACs（Question+CoT -> ADUs）的边界级 P/R/F1。
    DATA.json 中每个 item 需要至少包含：
      - id
      - Question
      - CoT
      - adus_text: List[str]  (gold ADUs, 覆盖 Question+CoT)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="DATA.json",
                    help="带 gold adus_text 的标注文件（list[dict] 或 JSONL）")
    ap.add_argument("--pdtb_csv", type=str, default="pdtb2.csv",
                    help="PDTB few-shot 数据路径（含 Arg1_RawText, Arg2_RawText, Connective_RawText）")
    ap.add_argument("--max_items", type=int, default=0,
                    help="最多评估多少条（0 或负数表示全部）")
    ap.add_argument("--dump_seg", type=str, default="seg.jsonl",
                    help="如果非空，则把 segmentation 结果写到这个 JSONL，文件名会自动加时间戳")
    args = ap.parse_args()

    # 运行时间戳（用于文件名、log）
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Few-shot block：使用手工设定的固定 few-shots，避免随机采样带来的不稳定
    shots = FIXED_FEW_SHOTS
    fewshot_block = render_fewshot_block(shots)

    client = make_client()
    items = load_items(args.input)
    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]

    print(f"[INFO] loaded {len(items)} items from {args.input}")

    # 处理 dump_seg 文件名，加上时间戳
    fout = None
    dump_seg_path = ""
    if args.dump_seg:
        base, ext = os.path.splitext(args.dump_seg)
        if not ext:
            ext = ".jsonl"
        dump_seg_path = f"{base}_{run_ts}{ext}"
        os.makedirs(os.path.dirname(dump_seg_path) or ".", exist_ok=True)
        fout = open(dump_seg_path, "w", encoding="utf-8")
        print(f"[INFO] segmentation debug will be written to: {dump_seg_path}")

    total_tp = total_fp = total_fn = 0
    total_gold_b = total_pred_b = 0

    # 存每个样本的 metrics，用于 best/worst case
    case_results: List[Dict[str, Any]] = []

    for idx, obj in enumerate(items):
        cid = obj.get("id")
        q = obj.get("Question") or ""
        cot = obj.get("CoT") or ""
        gold_adus = obj.get("adus_text") or []
        # 约定：最后一个 ADU 是 answer 信息，需要丢掉
        gold_adus = gold_adus[:-1]

        raw_full = f"{q} {cot}"
        full_text = preprocess(raw_full)

        if not gold_adus:
            print(f"[WARN] id={cid} 没有 adus_text，跳过。")
            continue

        gold_spans = align_gold_spans(full_text, gold_adus)
        if len(gold_spans) != len(gold_adus):
            print(
                f"[WARN] id={cid}: gold spans 数量({len(gold_spans)}) != adus_text 数量({len(gold_adus)})，"
                "可能有对齐问题。"
            )

        try:
            seg_res = segment_preprocessed_text(client, full_text, fewshot_block)
        except Exception as e:
            print(f"[ERROR] id={cid} segmentation error: {e}")
            continue

        pred_spans = extract_pred_spans(seg_res)
        metrics = compute_boundary_metrics(len(full_text), gold_spans, pred_spans)

        total_tp  += metrics["tp"]
        total_fp  += metrics["fp"]
        total_fn  += metrics["fn"]
        total_gold_b += metrics["gold_boundaries"]
        total_pred_b += metrics["pred_boundaries"]

        print(
            f"[CASE id={cid}] "
            f"GoldB={metrics['gold_boundaries']}, PredB={metrics['pred_boundaries']}, "
            f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        )

        case_results.append({
            "id": cid,
            "idx": idx,
            "metrics": metrics
        })

        if fout is not None:
            out_line = {
                "id": cid,
                "idx": idx,
                "full_text": full_text,
                "gold_adus": gold_adus,
                "gold_spans": gold_spans,
                "pred_segments": seg_res,
                "metrics": metrics,
            }
            fout.write(json.dumps(out_line, ensure_ascii=False) + "\n")

    if fout is not None:
        fout.close()

    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    rec  = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    print("\n========== Overall ACs Segmentation Metrics ==========")
    print(f"#Gold boundaries: {total_gold_b}")
    print(f"#Pred  boundaries: {total_pred_b}")
    print(f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # 计算 best / worst case（按 F1）
    best_case = None
    worst_case = None
    if case_results:
        best_case = max(case_results, key=lambda x: x["metrics"]["f1"])
        worst_case = min(case_results, key=lambda x: x["metrics"]["f1"])

        print("\n---------- Best Case ----------")
        print(f"id={best_case['id']}, "
              f"F1={best_case['metrics']['f1']:.4f}, "
              f"P={best_case['metrics']['precision']:.4f}, "
              f"R={best_case['metrics']['recall']:.4f}, "
              f"GoldB={best_case['metrics']['gold_boundaries']}, "
              f"PredB={best_case['metrics']['pred_boundaries']}")

        print("---------- Worst Case ----------")
        print(f"id={worst_case['id']}, "
              f"F1={worst_case['metrics']['f1']:.4f}, "
              f"P={worst_case['metrics']['precision']:.4f}, "
              f"R={worst_case['metrics']['recall']:.4f}, "
              f"GoldB={worst_case['metrics']['gold_boundaries']}, "
              f"PredB={worst_case['metrics']['pred_boundaries']}")

    print("======================================================")

    # ====== 将本次运行结果写入 log（带时间戳 + prompt） ======
    # few-shot 具体句子
    few_shots_serialized = [
        {"sentence": fs.sentence, "adus_text": fs.adus_text}
        for fs in shots
    ]

    log_entry: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_ts": run_ts,
        "model": MODEL_NAME,
        "input": args.input,
        "pdtb_csv": args.pdtb_csv,
        "max_items": args.max_items,
        "dump_seg": dump_seg_path,
        "overall": {
            "gold_boundaries": total_gold_b,
            "pred_boundaries": total_pred_b,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "few_shots": few_shots_serialized,
        "best_case": best_case,
        "worst_case": worst_case,
    }

    # log 文件名也带时间戳，防止覆盖
    log_path = f"acs_eval_runs_{run_ts}.log"
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            # 第一行写 PROMPT
            lf.write("PROMPT:\n")
            lf.write(INSTRUCTION.strip() + "\n\n")
            # 再写 JSON 数据（单行）
            lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"[INFO] 当前运行结果已写入日志: {log_path}")
    except Exception as e:
        print(f"[WARN] 写日志时出错: {e}")


if __name__ == "__main__":
    # for tmp in range(10):
    main()
