# CoT Logic Graph Inference

This repository packages a simple end-to-end inference pipeline and need `python >= 3.10`:

1. **Argument Component segmentation**: split a single input text into proposition-level ADUs  
   - `llm`: prompt-based splitter (Qwen via OpenAI-compatible API)
   - `unirst`: UniRST EDU segmentation + merge rules

2. **Argumentative relations identify and classification**: infer directed argumentative relations among ADUs using an LLM

Output is a single JSON file containing the input text, ADUs, and directed labeled edges.

---

## 1) Setup

### Python environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### API key

- Set environment variable `DASHSCOPE_API_KEY`
- Or pass `--api_key` at runtime

---

## 2) Run inference on a single txt

```bash
python scripts/infer_text.py \
  --input_txt data/imput.txt \
  --ac_method llm \
  --ac_model qwen3-max \
  --ari_model qwen3-max \
  --out_dir data/outputs
```

---

## 3) Output

- Explicit filename: `--out_name myrun.json`
- Template: `--name_template "{stem}_AC-{ac_method}_ARI-{aric_model}_{timestamp}.json"`


---

## 4) Customize ACs few-shot examples

Default few-shots:

- `data/fewshots/default.json`

Override:

```bash
python scripts/infer_text.py \
  --input_txt data/input.txt \
  --ac_method llm \
  --fewshot_json data/fewshots/my_fewshots.json
```

Few-shot JSON schema:

```json
[
  {
    "sentence": "your example sentence",
    "adus_text": ["adu1", "adu2"]
  }
]
```

---

## 5) Use UniRST + Merge for ACs

Install UniRST deps:

```bash
pip uninstall isanlp -y
pip install git+https://github.com/iinemo/isanlp.git
pip install -r requirements-unirst.txt
```

Run:

```bash
python scripts/infer_text.py \
  --input_txt data/input.txt \
  --ac_method unirst \
  --unirst_cuda_device -1
```

---

## 6) JSON format

Top-level fields:

- `text`: original input text
- `ac.adus_text`: list of ADUs (strings)
- `aric.edges`: list of directed labeled edges:
  - `src`, `tgt` (0-based ADU indices)
  - `label` in {Support, Attack, Restatement, Non-logical}
  - `confidence` in [0,1]
  - `evidence` with `type`, `connectives`, `rationale`

Use `--save_debug` to include intermediate artifacts (sentence-level outputs, raw model JSON).

---
## Notes

- Do not commit API keys. Use environment variables instead.
- Sentence splitting for `llm` mode is heuristic; UniRST mode is typically more stable for long texts.
