from typing import List, Dict, Any
from task_base import Task
import json, random, re

class TaskQA(Task):
    def __init__(self):
        with open("prompts/qa/qa_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/qa/qa_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        # Load sharded examples once
        try:
            with open("data/sharded_examples.json", "r", encoding='utf-8') as f:
                all_examples = json.load(f)
            # Filter examples by task
            self.examples = [ex for ex in all_examples if ex.get("task") == self.get_task_name()]
        except Exception as e:
            print(f"Warning: Could not load sharded examples: {e}")
            self.examples = []

        self.seed = 42
        random.seed(self.seed)

        self.answer_extraction_strategy = "gen"

    def get_dataset_file(self) -> str:
        return "data/qa/sharded_qa.json"

    def get_samples(self, filter="full"):
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        return data

    def get_task_name(self):
        return "qa"
        
    def get_answer_description(self) -> str:
        return "The answer should be a single letter (A, B, C, or D)."

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        # Ground truth answer (expected to be 'A', 'B', etc.)
        gold = sample["label"].strip().upper()

        try:
            # Extract only the first valid 'A', 'B', etc. from the model output
            # e.g., "The correct answer is A." -> "A"
            match = re.search(r"\b([AaBbCcDd])\b", extracted_answer)
            if not match:
                return {"score": 0.0, "error": f"No valid answer found in: {repr(extracted_answer)}"}
            extracted = match.group(1).upper()
        except Exception as e:
            return {"score": 0.0, "error": f"Answer could not be extracted: {repr(e)}"}

        # Compare model output with gold label
        score = 1.0 if extracted == gold else 0.0
        return {"score": score}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        #replace varibles except for fewshot examples
        return (
            self.fully_specified_prompt
            .replace("[[text]]", sample["text"])
            .replace("[[question]]", sample["question"])
            .replace("[[choice1]]", sample["choice1"])
            .replace("[[choice2]]", sample["choice2"])
            .replace("[[choice3]]", sample["choice3"])
            .replace("[[choice4]]", sample["choice4"])
        )

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        # Combine all shards into a newline-separated string
        concatenated_shards = "\n".join(shard["shard"] for shard in sample["shards"])

        # Prepend the system message - fewshot examples not yet handled
        return f"{self.system_prompt.strip()}\n\n{concatenated_shards.strip()}"

    def populate_full_examples(self, num_examples: int = 5) -> str:
        """Generate few-shot examples for full prompts."""
        try:
            # Use pre-loaded examples
            examples = self.examples[:num_examples]
            
            # Load template
            with open(f"prompts/{self.get_task_name()}/{self.get_task_name()}_full_example.txt", "r", encoding='utf-8') as f:
                template = f.read()
            
            # Format each example
            formatted_examples = []
            for example in examples:
                example_data = {
                    "text": example["text"],
                    "question": example["question"],
                    "choice1": example["choice1"],
                    "choice2": example["choice2"],
                    "choice3": example["choice3"],
                    "choice4": example["choice4"],
                    "label": example["label"]
                }
                formatted_example = template
                for key, value in example_data.items():
                    formatted_example = formatted_example.replace(f"[[{key}]]", str(value))
                formatted_examples.append(formatted_example)
            
            return "\n".join(formatted_examples)
        
        except Exception as e:
            print(f"Warning: Could not format examples: {e}")
            return ""

    def populate_sharded_examples(self, num_examples: int = 5) -> str:
        """Generate few-shot examples for sharded/concat prompts."""
        try:
            # Use pre-loaded examples
            examples = self.examples[:num_examples]
            
            # Load template
            with open("prompts/sharded_example.txt", "r", encoding='utf-8') as f:
                template = f.read()
            
            # Format each example
            formatted_examples = []
            for example in examples:
                # Concatenate all shards
                shards_text = "\n".join([shard["shard"] for shard in example["shards"]])
                
                # Replace placeholders
                formatted_example = template.replace("[[shards]]", shards_text)
                formatted_example = formatted_example.replace("[[label]]", example["label"])
                formatted_examples.append(formatted_example)
            
            return "\n".join(formatted_examples)
        
        except Exception as e:
            print(f"Warning: Could not format examples: {e}")
            return ""

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        # im not sure what this does
        return response["answer"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process QA sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            # "question": sample["question"],
            # "answer": sample["answer"],
            # maybe we'll have (for the sample): task_id (e.g sharded-QA/1246), task, talata, question, choice1, ... , choice4, shards, label/answer
            # or we can have variables grouped 
        }

if __name__ == "__main__":
    # Initialize the task (this automatically loads the system and full prompts)
    task = TaskQA()

    # Example sample
    sample = {
        "task_id": "sharded-qa/1",
        "text": "Ang nayon ng Haldarsvík ay may kakaibang simbahang oktagonal.",
        "question": "Saan matatagpuan ang mga eskultura ng kalapati?",
        "choice1": "Kalahating oras mula sa nayon",
        "choice2": "Sa bakuran ng simbahan",
        "choice3": "Sa isla ng Eysturoy",
        "choice4": "Sa loob ng simbahan",
        "label": "B",
        "shards": [
            {
                "shard_id": 1,
                "shard": "Saan makikita ang mga eskultura ng kalapati?"
            },
            {
                "shard_id": 2,
                "shard": "Sa bakuran ng simbahan ng HaldarsvÃ­k, may mga marmol na eskultura ng mga kalapati na makikita sa ilan sa mga puntod."
            },
            {
                "shard_id": 3,
                "shard": "Makikita sa nayon ng HaldarsvÃ­k ang magagandang tanawin ng katabing isla na Eysturoy."
            },
            {
                "shard_id": 4,
                "shard": "Ang nayon ng HaldarsvÃ­k ay kilala rin sa kakaibang simbahang hugis oktagon."
            },
            {
                "shard_id": 5,
                "shard": "Ang pamamasyal sa pook na ito ay aabutin ng mga kalahating oras."
            },
            {
                "shard_id": 6,
                "shard": "Piliin ang A kung ang lugar na kalahating oras ang layo mula sa nayon."
            },
            {
                "shard_id": 7,
                "shard": "Isa sa pagpipilian ay ang B kung ang bakuran mismo ng simbahan."
            },
            {
                "shard_id": 8,
                "shard": "Piliin ang titik C kung matatagpuan ang mga eskultura sa isla ng Eysturoy."
            },
            {
                "shard_id": 9,
                "shard": "Piliin ang D kung nasa loob ng simbahan ang mga eskulturang ito."
            }
        ],
        "task": "qa"
    }

    # # Test populate_fully_specific_prompt
    # print("=== Fully Specific Prompt ===")
    # print(task.populate_fully_specific_prompt(sample))
    # print("\n")

    # # Test populate_concat_prompt
    # print("=== Concatenated Prompt ===")
    # print(task.populate_concat_prompt(sample))

    print("=" * 80)
    print("TESTING FEW-SHOT EXAMPLE FUNCTIONS")
    print("=" * 80)
    
    # Test 1: Full examples (for non-sharded prompts)
    print("\n" + "-" * 80)
    print("TEST 1: populate_full_examples()")
    print("-" * 80)
    try:
        full_examples = task.populate_full_examples(num_examples=5)
        if full_examples:
            print(f"✓ Successfully loaded {full_examples.count('[[label]]') if '[[label]]' in full_examples else 'multiple'} full examples")
            print(f"\nFirst 1000 characters:\n{full_examples[:1000]}...")
            print(f"\nTotal length: {len(full_examples)} characters")
        else:
            print("✗ No examples returned (empty string)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Sharded examples (for concat/shuffle-concat prompts)
    print("\n" + "-" * 80)
    print("TEST 2: populate_sharded_examples()")
    print("-" * 80)
    try:
        sharded_examples = task.populate_sharded_examples(num_examples=5)
        if sharded_examples:
            print(f"✓ Successfully loaded {sharded_examples.count('[[label]]') if '[[label]]' in sharded_examples else 'multiple'} sharded examples")
            print(f"\nFirst 1000 characters:\n{sharded_examples[:1000]}...")
            print(f"\nTotal length: {len(sharded_examples)} characters")
        else:
            print("✗ No examples returned (empty string)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Integration test - check if examples insert into prompts correctly
    print("\n" + "-" * 80)
    print("TEST 3: Integration with prompts")
    print("-" * 80)
    
    # Test with full prompt
    full_prompt = task.populate_fully_specific_prompt(sample)
    if "[[fewshot_examples]]" in full_prompt:
        print("✓ Full prompt contains [[fewshot_examples]] placeholder")
        full_examples = task.populate_full_examples(num_examples=5)
        final_prompt = full_prompt.replace("[[fewshot_examples]]", full_examples)
        print(f"✓ Replaced placeholder successfully")
        print(f"  Before: {len(full_prompt)} chars")
        print(f"  After:  {len(final_prompt)} chars")
        print(f"  Added:  {len(final_prompt) - len(full_prompt)} chars")
    else:
        print("✗ Full prompt does not contain [[fewshot_examples]] placeholder")
    
    # Test with concat prompt
    concat_prompt = task.populate_concat_prompt(sample)
    if "[[fewshot_examples]]" in concat_prompt:
        print("\n✓ Concat prompt contains [[fewshot_examples]] placeholder")
        sharded_examples = task.populate_sharded_examples(num_examples=5)
        final_prompt = concat_prompt.replace("[[fewshot_examples]]", sharded_examples)
        print(f"✓ Replaced placeholder successfully")
        print(f"  Before: {len(concat_prompt)} chars")
        print(f"  After:  {len(final_prompt)} chars")
        print(f"  Added:  {len(final_prompt) - len(concat_prompt)} chars")
    else:
        print("✗ Concat prompt does not contain [[fewshot_examples]] placeholder")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)