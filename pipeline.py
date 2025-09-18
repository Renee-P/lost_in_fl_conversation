# src/pipeline.py
from dataclasses import dataclass
from pathlib import Path
import json, re
#GPTed DO NOT TRUST (yet)
@dataclass
class PromptTemplates:
    system: str
    user_template: str  # contains [[EXAMPLES]] and [[INSTRUCTION]] etc.

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def render(template: str, **kwargs) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace(f"[[{k}]]", v)
    return out

class ModelClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
    def call(self, system: str, user: str) -> str:
        # TODO: wire to your provider (OpenAI/Bedrock/etc.)
        # Must return a JSON string: {"segments":[{"segment":"..."}, ...]}
        raise NotImplementedError

class Segmenter:
    def __init__(self, templates: PromptTemplates, client: ModelClient, min_segments:int=3):
        self.templates = templates
        self.client = client
        self.min_segments = min_segments

    def segment(self, instruction: str, examples_block: str) -> dict:
        user = render(self.templates.user_template,
                      EXAMPLES=examples_block,
                      INSTRUCTION=instruction,
                      MIN_SEGMENTS=str(self.min_segments))
        raw = self.client.call(self.templates.system, user)
        obj = json.loads(raw)
        segs = [s["segment"] for s in obj.get("segments", [])]
        if len(segs) < self.min_segments:
            return {"ok": False, "reason": "lt_min_segments", "segments": segs}
        # Optional: dedupe, trim, and assert non-overlap heuristics
        return {"ok": True, "segments": segs}

def run_batch(data_path: str, out_path: str, examples_path: str, tmpl_dir: str, model_name: str):
    sys_p = Path(tmpl_dir)/"segmentation_system_prompt.txt"
    usr_p = Path(tmpl_dir)/"segmentation_full_prompt.txt"
    templates = PromptTemplates(system=load_text(sys_p), user_template=load_text(usr_p))
    client = ModelClient(model_name)
    segmenter = Segmenter(templates, client, min_segments=3)

    examples_block = load_text(Path(examples_path))   # a file containing your few-shot examples
    with open(data_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            res = segmenter.segment(rec["instruction"], examples_block)
            out = {"id": rec.get("id"), **res}
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
