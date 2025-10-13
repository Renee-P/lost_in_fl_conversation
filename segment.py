import os
import json
from utils.model import generate_json  
import re
from config import config   # for testing

# TO-DO: Handle AS

class Segmentation:
    def __init__(self, config):
        self.config = config  

    def process_task(self, task):
        data_path = os.path.join(self.config["raw_data_dir"], task, self.config["raw_file_names"][task])
        prompt_path = os.path.join(self.config["prompts_dir"], task, f"{task}_segment.txt")

        with open(prompt_path, encoding="utf-8") as f:
            prompt_template = f.read()

        segmented_instances = []
        with open(data_path, encoding="utf-8") as fin:
            for line in fin:
                instance = json.loads(line)
                instance_vars = self._extract_placeholders(instance, task, prompt_template)
                prompt = self._fill_prompt(prompt_template, instance_vars)
                messages = [{"role": "user", "content": prompt}]
                try:
                    segments = generate_json(messages)
                    if len(segments["segments"]) < 3:
                        continue  
                    instance["segments"] = segments["segments"]
                    segmented_instances.append(instance)
                except Exception as e:
                    print(f"Error on instance {instance.get('id', '')}: {e}")
        return segmented_instances

    def _extract_placeholders(self, instance, task, template):
        pr = instance["prompts"][0]
        placeholders = re.findall(r"\[\[(.*?)\]\]", template)
        result = {}
        for key in placeholders:
            if key in pr:
                result[key] = pr[key]
            else:
                print(f"Warning: '{key}' not found in instance['prompts'][0] for instance id {instance.get('id', '')}")
                result[key] = ""
        return result

    def _fill_prompt(self, template, variables):
        for k, v in variables.items():
            template = template.replace(f"[[{k}]]", v)
        return template

if __name__ == "__main__":
    # testing 1 instance

    task = "cr"
    data_path = os.path.join(config["raw_data_dir"], task, config["raw_file_names"][task])
    prompt_path = os.path.join(config["prompts_dir"], task, f"{task}_segment.txt")

    with open(prompt_path, encoding="utf-8") as f:
        prompt_template = f.read()

    with open(data_path, encoding="utf-8") as fin:
        first_line = next(fin)
        instance = json.loads(first_line)

    seg = Segmentation(config=config)

    # Step 1: Test placeholder extraction
    instance_vars = seg._extract_placeholders(instance, task=task, template=prompt_template)
    # print("Extracted instance_vars:")
    # print(instance_vars)

    # Step 2: Test prompt filling
    filled_prompt = seg._fill_prompt(prompt_template, instance_vars)
    print("\nFilled Prompt:")
    print(filled_prompt)

    # You can extend this by mocking generate_json if you want to see how a segments structure would append.
    # note test the whole thing later - check the responses and adjust model/prompts
    # also test per task what the output is/looks like

    # tests a task and shows first instance result?
    # seg = Segmentation(config=config)
    # segment_results = seg.process_task("qa")
    # print(f"Processed {len(segment_results)} qa instances.")
    # print("First segmented instance (truncated):")
    # print(json.dumps(segment_results[0], ensure_ascii=False, indent=2))