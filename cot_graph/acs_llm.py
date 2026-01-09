from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from .fewshots import FewShot, render_fewshot_block
from .text_utils import normalize_ws, split_sentences
from .llm_client import make_openai_client

MODEL_DEFAULT = "qwen3-max"
TEMPERATURE_DEFAULT = 0.2
TOP_P_DEFAULT = 0.3
MAX_RETRIES_DEFAULT = 3
REQUEST_SLEEP_SEC_DEFAULT = 0.2

WORD = r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*"

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

def _looks_truncated(token: str) -> bool:
    return len(token) <= 2 or not re.fullmatch(WORD, token)

def _is_punct_only_text(s: str) -> bool:
    return bool(re.fullmatch(r"\s*[\W_]+\s*", s or ""))

def clean_llm_adus(sentence: str, adus_text: List[str]) -> List[str]:
    """
    Keep only plausible ADUs that appear in the sentence (whitespace-tolerant),
    and filter obvious truncations.
    """
    out: List[str] = []
    for t in adus_text:
        if not t or _is_punct_only_text(t):
            continue
        t = t.strip()

        if t not in sentence:
            pat_ws = re.escape(t).replace(r"\ ", r"\s+")
            if not re.search(pat_ws, sentence, flags=re.I):
                continue

        tail = re.findall(WORD, t)
        tail_bad = (len(tail) == 0) or _looks_truncated(tail[-1])
        starts_bad = bool(re.match(
            r"^(,|\band\b|\bor\b|\bbut\b|\bnor\b|\bas\b|\bwhich\b|\bthat\b|\bthen\b)\b",
            t.strip(), flags=re.I
        ))
        ends_bad = bool(re.search(
            r"(,|\bof\b|\bto\b|\bfrom\b|\bfor\b|\bwith\b|\bby\b|\bif\b|\bwhen\b|\bwhile\b|\bbecause\b)\s*$",
            t.strip(), flags=re.I
        ))
        if tail_bad or starts_bad or ends_bad:
            continue

        out.append(t)

    # de-dup, keep order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def call_llm_sentence(*, client, sentence: str, fewshot_block: str,
                      model: str = MODEL_DEFAULT,
                      temperature: float = TEMPERATURE_DEFAULT,
                      top_p: float = TOP_P_DEFAULT,
                      max_retries: int = MAX_RETRIES_DEFAULT) -> List[str]:
    user_prompt = (
        INSTRUCTION + "\n\n"
        "Few-shot examples:\n" + fewshot_block + "\n\n"
        "Now segment this sentence and output ONLY JSON with key 'adus_text':\n" + sentence
    )

    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise clause splitter. Only output valid JSON."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"},
            )
            out = (resp.choices[0].message.content or "").strip()
            data = _extract_json(out)
            texts = data.get("adus_text", [])
            if not isinstance(texts, list):
                raise ValueError("bad 'adus_text'")
            clean = [str(x).strip() for x in texts if isinstance(x, str) and x.strip()]
            return clean_llm_adus(sentence, clean) or []
        except Exception as e:
            last_err = e
            time.sleep(0.35)
            continue

    return []


def segment_text_llm(
    text: str,
    *,
    base_url: str,
    api_key: Optional[str],
    model: str = MODEL_DEFAULT,
    fewshots: List[FewShot],
    temperature: float = TEMPERATURE_DEFAULT,
    top_p: float = TOP_P_DEFAULT,
    request_sleep_sec: float = REQUEST_SLEEP_SEC_DEFAULT,
) -> Dict[str, Any]:
    """
    Segment a single text into ADUs using an LLM clause splitter.
    """
    t = normalize_ws(text)
    if not t:
        return {"method": "llm", "adus_text": [], "sentences": [], "n_api_calls": 0}

    client = make_openai_client(base_url=base_url, api_key=api_key, timeout_sec=180.0)
    fewshot_block = render_fewshot_block(fewshots)

    sents = split_sentences(t)

    sentences_out: List[Dict[str, Any]] = []
    n_calls = 0
    all_adus: List[str] = []

    for i, (s, e, sent) in enumerate(sents):
        raw_adus = call_llm_sentence(
            client=client,
            sentence=sent,
            fewshot_block=fewshot_block,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        n_calls += 1

        adus_global = []
        for t_adu in raw_adus:
            idx = sent.find(t_adu)
            if idx >= 0:
                adus_global.append({"span": [s + idx, s + idx + len(t_adu)], "text": t_adu})
            else:
                adus_global.append({"span": None, "text": t_adu})
            all_adus.append(t_adu)

        sentences_out.append({
            "sent_id": i,
            "span": [s, e],
            "text": sent,
            "adus": adus_global,
        })

        time.sleep(request_sleep_sec)

    return {"method": "llm", "adus_text": all_adus, "sentences": sentences_out, "n_api_calls": n_calls}
