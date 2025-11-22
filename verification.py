"""
Verification stage:
- Reads JSONL records with fields:
    id
    segments               # list[str] OR list[{"segment": "..."}]
    rephrased_segments     # list[str] OR list[{"segment": "..."}]
- Calls the verifier LLM using prompts/verification.txt
- Writes JSONL with:
    id
    verified_rephrased_segments  # list[str] that passed BOTH checks
    verdicts                     # full list of per-pair verdicts
"""

from pathlib import Path
import json
import argparse
from typing import List, Dict, Any

# Reuse the same wrappers/paths you use elsewhere in the repo
from config import PROMPTS_DIR  # keep consistent with segmentation/rephrasing
# Your LLM wrapper used elsewhere (e.g., in segmentation/rephrasing)
from utils.model import call_llm  # adjust if your wrapper has a different name/signature


VERIFY_PROMPT_PATH = Path(PROMPTS_DIR) / "verification.txt"


def _load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _unwrap_segments(xs: List[Any]) -> List[str]:
    """
    Accept either:
      - ["seg1", "seg2", ...]
      - [{"segment": "seg1"}, {"segment": "seg2"}, ...]
    and normalize to list[str].
    """
    out: List[str] = []
    for x in xs:
        if isinstance(x, dict) and "segment" in x:
            out.append(str(x["segment"]))
        else:
            out.append(str(x))
    return out


def _render_pairs_json(original: List[str], rephrased: List[str]) -> str:
    """
    Produce the 'pairs' JSON payload expected by verification.txt.
    Truncates to the shorter length if sizes differ.
    """
    n = min(len(original), len(rephrased))
    pairs = [{"original_segment": original[i], "rephrased_segment": rephrased[i]} for i in range(n)]
    return json.dumps({"pairs": pairs}, ensure_ascii=False)


def _verify_pairs(pairs_json: str) -> List[Dict[str, Any]]:
    """
    Calls the LLM with the verification prompt and returns a list of verdict dicts.
    The prompt strictly requires a single JSON object containing "verdicts".
    """
    prompt = _load_text(VERIFY_PROMPT_PATH).replace("[[PAIRS]]", pairs_json)

    # If your wrapper supports a system message, you can pass a short one like "You are a verifier."
    # The repo you showed keeps prompts as user-only; we follow that here.
    raw = call_llm(system=None, user=prompt)

    # Be defensive in case the model prepends/appends stray text.
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start:end + 1])
        else:
            raise

    verdicts = obj.get("verdicts", [])
    # normalize verdict fields
    norm = []
    for v in verdicts:
        norm.append({
            "preserves_meaning": bool(v.get("preserves_meaning", False)),
            "is_actionable": bool(v.get("is_actionable", False)),
            "reason": str(v.get("reason", "")).strip()
        })
    return norm


def verify_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifies a single JSONL record.
    """
    original = _unwrap_segments(rec.get("segments", []))
    rephrased = _unwrap_segments(rec.get("rephrased_segments", []))

    pairs_json = _render_pairs_json(original, rephrased)
    verdicts = _verify_pairs(pairs_json)

    kept = [
        shard for shard, v in zip(rephrased, verdicts)
        if v.get("preserves_meaning") and v.get("is_actionable")
    ]

    return {
        "id": rec.get("id"),
        "verified_rephrased_segments": kept,
        "verdicts": verdicts
    }


def run_verification(in_path: str, out_path: str) -> None:
    """
    Stream JSONL -> JSONL to match your other stages.
    """
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = verify_record(rec)
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="outputs/rephrased.jsonl")
    ap.add_argument("--out", dest="out_path", default="outputs/verified_shards.jsonl")
    args = ap.parse_args()
    run_verification(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
