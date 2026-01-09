from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List

import httpx
import certifi

try:
    from openai import OpenAI
    from openai import APITimeoutError, APIConnectionError, APIStatusError, RateLimitError
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    APITimeoutError = APIConnectionError = APIStatusError = RateLimitError = Exception  # type: ignore


DEFAULT_MODEL = "qwen3-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

LABELS = {
    "Support": "ADU A provides justification, evidence, or reasoning that strengthens ADU B.",
    "Attack": "ADU A directly denies, contradicts, or rejects the content or correctness of ADU B.",
    "Restatement": "ADU A rephrases or repeats the content of ADU B without adding new constraints.",
    "Non-logical": "ADU A and ADU B are related in a non-argumentative way (e.g., pure background, meta-comment) and A is not used as a reason to accept/reject B.",
    "No-Relation": "No clear argumentative relation between ADU A and ADU B."
}

LABEL_CANON = {lbl.lower(): lbl for lbl in LABELS.keys()}

def canonical_relation(rel: str) -> str:
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
    allowed = {"Support", "Attack", "Restatement", "Non-logical", "No-Relation"}
    for a in allowed:
        if r == a.lower():
            return a
    return ""


def _norm_idx(v, n):
    try:
        i = int(v)
        if 0 <= i < n:
            return i
    except Exception:
        pass
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
            trust_env=False,
        )
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        self.timeout_sec = timeout_sec

    def _request_with_retry(self, *, model, messages, response_format, max_retries=6):
        delay = 1.0
        backoff = 1.6
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    top_p=0.3,
                    response_format=response_format,
                    timeout=self.timeout_sec,
                )
            except (APITimeoutError, APIConnectionError, httpx.ReadTimeout, httpx.ConnectTimeout):
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay + random.random() * 0.5)
                delay *= backoff
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay + random.random() * 0.7)
                delay *= backoff
            except APIStatusError as e:
                code = getattr(e, "status_code", 500)
                if code >= 500 or code in (408, 409, 429, 499, 502, 503, 504):
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay + random.random() * 0.7)
                    delay *= backoff
                else:
                    raise

        raise RuntimeError("Retries exhausted")

    def call_ari(self, adus: List[str], model: str) -> Dict[str, Any]:
        n = len(adus)
        adus_block = "\n".join([f"[{i}] {adus[i]}" for i in range(n)])

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

ADUs:
{adus_block}

Return STRICT JSON only.
""".strip()

        messages = [
            {"role": "system", "content": "You are an expert argument mining system. Only output valid JSON."},
            {"role": "user", "content": user},
        ]

        resp = self._request_with_retry(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )

        out = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(out)
        except Exception:
            import re
            m = re.search(r"\{.*\}", out, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}


def infer_aric(
    adus_text: List[str],
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    adus_text = [str(x) for x in (adus_text or []) if str(x).strip()]
    n = len(adus_text)
    if n == 0:
        return {"edges": []}

    qwen = QwenClient(base_url=base_url, api_key=api_key)
    raw = qwen.call_ari(adus=adus_text, model=model)

    edges_in = raw.get("edges", [])
    if not isinstance(edges_in, list):
        edges_in = []

    edges_out = []
    for e in edges_in:
        if not isinstance(e, dict):
            continue
        src = _norm_idx(e.get("src"), n)
        tgt = _norm_idx(e.get("tgt"), n)
        if src is None or tgt is None or src == tgt:
            continue
        lbl = canonical_relation(e.get("label", ""))
        if not lbl or lbl == "No-Relation":
            continue
        edges_out.append({
            "src": src,
            "tgt": tgt,
            "label": lbl,
            "confidence": _norm_conf(e.get("confidence", 0.0)),
            "evidence": _norm_evidence(e.get("evidence", {})),
        })

    return {"edges": edges_out, "raw": raw}
