# print_edus.py
import json
import argparse
import re

WS_RE = re.compile(r"\s+")

def norm(s: str) -> str:
    return WS_RE.sub(" ", (s or "")).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="case_edus.jsonl")
    ap.add_argument("--id", type=int, default=None, help="Only print this case id (optional).")
    ap.add_argument("--show_head", action="store_true", help="Print eval_text_head before EDUs.")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id", None)
            if args.id is not None and cid != args.id:
                continue

            domain = obj.get("domain", "")
            if args.show_head:
                print(f"=== id={cid} domain={domain} ===")
                print(norm(obj.get("eval_text_head", "")))
                print("")

            edus = obj.get("pred_edus", [])
            print(f"--- Pred EDUs (id={cid}, domain={domain}, n={len(edus)}) ---")
            for i, edu in enumerate(edus):
                s = edu.get("start")
                e = edu.get("end")
                t = norm(edu.get("text", ""))
                print(f"{i:03d}\t[{s},{e}]\t{t}")
            print("")

if __name__ == "__main__":
    main()
