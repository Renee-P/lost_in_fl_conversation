from typing import List, Dict, Any
from task_base import Task
import json, random, re # not sure

class TaskTD(Task):
    def __init__(self):
        with open("prompts/td/td_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/td/td_system_prompt.txt", "r") as f:
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

    def get_task_name(self) -> str:
        return "td"

    def get_dataset_file(self) -> str:
        return "data/td/sharded_td.json"

    def get_samples(self, filter="full") -> List[Dict[str, Any]]:
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        return data

    def get_answer_description(self) -> str:
        return "The answer should be \"Malinis\" or \"Mapoot\"."

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        # Ground truth answer (expected to be 'malinis' or 'mapoot')
        gold = sample["label"].strip().lower()

        try:
            # Extract only the first valid 'malinis' or 'mapoot' from the model output
            # e.g., "Ang sagot ay malinis." -> "malinis"
            match = re.search(r"\b(malinis|mapoot)\b", extracted_answer, re.IGNORECASE)
            if not match:
                return {"score": 0.0, "error": f"No valid answer found in: {repr(extracted_answer)}"}
            extracted = match.group(1).lower()
        except Exception as e:
            return {"score": 0.0, "error": f"Answer could not be extracted: {repr(e)}"}

        # Compare model output with gold label
        score = 1.0 if extracted == gold else 0.0
        return {"score": score}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return (
            self.fully_specified_prompt
            .replace("[[text]]", sample["text"])
        )

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Concatenate shards for the PI task and prepend the system prompt.
        [[fewshot_examples]] is ignored here (handled elsewhere).
        """
        concatenated_shards = "\n".join(shard["shard"] for shard in sample["shards"])
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

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Given a sample in the dataset file, return a dictionary with all the information from the original sample; helpful for displaying the sample in the annotation UI"""
        pass

if __name__ == "__main__":
    sample = {
       "task_id": "sharded-td/1",
       "text": "tanggap ko na talo si Mar Roxas pero sana naman si LENI ROBREDO ay manalo. #LabanLeni",
       "label": "Malinis",
       "shards": [
            {
                "shard_id": 1,
                "shard": "Malinis o Mapoot ang tweet?"
            },
            {
                "shard_id": 2,
                "shard": "Sinabi sa tweet na 'tanggap ko na talo si Mar Roxas'."
            },
            {
                "shard_id": 3,
                "shard": "Kahit daw matalo si Mar Roxas, 'sana naman si LENI ROBREDO ay manalo'."
            },
            {
                "shard_id": 4,
                "shard": "May hashtag na #LabanLeni."
            }
       ],
       "task": "td"
    }

    task = TaskTD()
    # print("FULL PROMPT:\n" + task.populate_fully_specific_prompt(sample))
    # print("\nCONCAT PROMPT:\n" + task.populate_concat_prompt(sample))

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
