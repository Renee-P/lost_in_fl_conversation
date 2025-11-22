from typing import List, Dict, Any
from task_base import Task
import json, random, re 
from sacrebleu.metrics import CHRF

class TaskMT(Task):
    def __init__(self):
        with open("prompts/mt/mt_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/mt/mt_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()

        # Load sharded examples once
        try:
            with open("data/sharded_mt_examples.json", "r", encoding='utf-8') as f:
                all_examples = json.load(f)
            # Filter examples by task
            self.examples = [ex for ex in all_examples if ex.get("task") == self.get_task_name()]
        except Exception as e:
            print(f"Warning: Could not load sharded examples: {e}")
            self.examples = []

        self.seed = 42
        random.seed(self.seed)

        self.answer_extraction_strategy = "full_response"

    def get_task_name(self) -> str:
        return "mt"

    def get_dataset_file(self) -> str:
        return "data/sharded_mt.json"

    def get_samples(self, filter="full") -> List[Dict[str, Any]]:
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        return data

    def get_answer_description(self) -> str:
        return "The answer should be the translation of the input text from the source language to the target language."

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        gold = sample["label"].strip()

        try:
            cleaned_answer = re.sub(r'^Salin:\s*', '', extracted_answer, flags=re.IGNORECASE).strip()

            if not cleaned_answer:
                return {
                    "score": 0.0,
                    "error": f"Empty answer after cleaning: {repr(extracted_answer)}"
                }

            chrf = CHRF(word_order=2)
            chrf_score = chrf.sentence_score(cleaned_answer, [gold]).score / 100.0
            # print(f"DEBUG - ChrF++ score: {chrf_score}")

            return {
                "score": chrf_score,
            }

        except Exception as e:
            return {
                "score": 0.0,
                "error": f"Error computing ChrF++: {repr(e)}"
            }

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return (
            self.fully_specified_prompt
            .replace("[[language]]", sample["language"])
            .replace("[[text]]", sample["text"])
        )

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Concatenate shards and prepend the system prompt.
        [[fewshot_examples]] is ignored here (handled elsewhere).
        """
        concatenated_shards = "\n".join(shard["shard"] for shard in sample["shards"])
        prompt = f"{self.system_prompt.strip()}\n\n{concatenated_shards.strip()}"
        return (
            prompt
            .replace("[[language]]", sample["language"])
        )

    def populate_sharded_prompt(self, sample, turn_index):
        shards = sample["shards"]

        # Turn 0 → reveal shard 0
        if turn_index == 0:
            shard = shards[0]
            return shard["shard"], shard["shard_id"], 0.0

        # Turn 1 to N → reveal shard turn_index
        elif turn_index < len(shards):
            shard = shards[turn_index]
            return shard["shard"], shard["shard_id"], 0.0

        # Beyond shards → no more
        else:
            return None, -1, 0.0

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
                # Use text and label fields directly
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
        "task_id": "sharded-mt/1",
        "label": "They include the Netherlands, with Anna Jochemsen finishing ninth in the women's standing class in the Super-G yesterday, and Finland with Katja Saarinen finishing tenth in the same event.",
        "text": "Kasama dito ang Netherlands, dahil si Anna Jochemsen ay nagtapos na ikasiyam sa klaseng pambabaeng nakatayo sa Super-G kahapon, at ang Finland dahil si Katja Saarinen ay nagtapos na ikasampu sa parehong laban.",
        "language": "English",
        "shards": [
            {
                "shard_id": "1",
                "shard": "Isinasalin mo ang isang tekstong Ingles tungo sa Filipino habang ito ay ini-transcribe nang live. Ibibigay ko sa iyo ang teksto nang paunti-unti. Sa bawat turn, dapat mong ibalik ang salin ng BUONG TEKSTO (hindi lamang ang huling bahagi). Dapat mong isaalang-alang ang lahat ng bahagi kapag nagsasalin, at hindi lamang ang pinakahuli. Maaari mo ring baguhin ang mga naunang salin kung sa tingin mo ay may pagkakamali ka. Unang Bahagi: So many of us find ourselves watching a television show"
            },
            {
                "shard_id": "2",
                "shard": "Natatanggap ko na ang pinakabagong bahagi ng dokumento. Paki-salin ang buong dokumento hanggang sa puntong ito. Isa pang bahagi: that informs us of a process or experience"
            },
            {
                "shard_id": "3",
                "shard": "Natatanggap ko na ang pinakabagong bahagi ng dokumento. Paki-salin ang buong dokumento hanggang sa puntong ito. Isa pang bahagi: in which we will never participate or apply that knowledge."
            }
        ]
    }

    task = TaskMT()
    
    print("=" * 60)
    print("TESTING TaskMT")
    print("=" * 60)
    
    # Test prompts
    print("\nFULL PROMPT:")
    print("-" * 60)
    print(task.populate_fully_specific_prompt(sample))
    
    print("\n\nCONCAT PROMPT:")
    print("-" * 60)
    print(task.populate_concat_prompt(sample))
    
    # Test evaluator function
    print("\n\n" + "=" * 60)
    print("TESTING EVALUATOR")
    print("=" * 60)
    
    # Test case 1: Perfect match
    test_answer_1 = "They include the Netherlands, with Anna Jochemsen finishing ninth in the women's standing class in the Super-G yesterday, and Finland with Katja Saarinen finishing tenth in the same event."
    result_1 = task.evaluator_function(test_answer_1, sample)
    print(f"\nTest 1 - Perfect match:")
    print(f"Answer: {test_answer_1}")
    print(f"Result: {result_1}")
    
    # Test case 2: With "Salin" prefix
    test_answer_2 = "Salin: They include the Netherlands, with Anna Jochemsen finishing ninth in the women's standing class in the Super-G yesterday, and Finland with Katja Saarinen finishing tenth in the same event."
    result_2 = task.evaluator_function(test_answer_2, sample)
    print(f"\nTest 2 - With 'Salin:' prefix:")
    print(f"Answer: {test_answer_2}")
    print(f"Result: {result_2}")
    
    # Test case 3: Partial match
    test_answer_3 = "They include Netherlands with Anna finishing ninth in the women's standing class in the Super-G yesterday, and Finland with Katja finishing tenth in the same event."
    result_3 = task.evaluator_function(test_answer_3, sample)
    print(f"\nTest 3 - Partial match:")
    print(f"Answer: {test_answer_3}")
    print(f"Result: {result_3}")
    
    # Test case 4: Empty answer
    test_answer_4 = ""
    result_4 = task.evaluator_function(test_answer_4, sample)
    print(f"\nTest 4 - Empty answer:")
    print(f"Answer: '{test_answer_4}'")
    print(f"Result: {result_4}")
    
    # Test case 5: Only "BUOD:" (should error)
    test_answer_5 = "SALIN: "
    result_5 = task.evaluator_function(test_answer_5, sample)
    print(f"\nTest 5 - Only 'SALIN:' prefix:")
    print(f"Answer: '{test_answer_5}'")
    print(f"Result: {result_5}")
    
    # print("\n" + "=" * 60)

    # print("=" * 80)
    # print("TESTING FEW-SHOT EXAMPLE FUNCTIONS")
    # print("=" * 80)
    
    # # Test 1: Full examples (for non-sharded prompts)
    # print("\n" + "-" * 80)
    # print("TEST 1: populate_full_examples()")
    # print("-" * 80)
    # try:
    #     full_examples = task.populate_full_examples(num_examples=5)
    #     if full_examples:
    #         print(f"✓ Successfully loaded {full_examples.count('[[label]]') if '[[label]]' in full_examples else 'multiple'} full examples")
    #         print(f"\nFirst 1000 characters:\n{full_examples[:1000]}...")
    #         print(f"\nTotal length: {len(full_examples)} characters")
    #     else:
    #         print("✗ No examples returned (empty string)")
    # except Exception as e:
    #     print(f"✗ Error: {e}")
    
    # # Test 2: Sharded examples (for concat/shuffle-concat prompts)
    # print("\n" + "-" * 80)
    # print("TEST 2: populate_sharded_examples()")
    # print("-" * 80)
    # try:
    #     sharded_examples = task.populate_sharded_examples(num_examples=5)
    #     if sharded_examples:
    #         print(f"✓ Successfully loaded {sharded_examples.count('[[label]]') if '[[label]]' in sharded_examples else 'multiple'} sharded examples")
    #         print(f"\nFirst 1000 characters:\n{sharded_examples[:1000]}...")
    #         print(f"\nTotal length: {len(sharded_examples)} characters")
    #     else:
    #         print("✗ No examples returned (empty string)")
    # except Exception as e:
    #     print(f"✗ Error: {e}")
    
    # # Test 3: Integration test - check if examples insert into prompts correctly
    # print("\n" + "-" * 80)
    # print("TEST 3: Integration with prompts")
    # print("-" * 80)
    
    # # Test with full prompt
    # full_prompt = task.populate_fully_specific_prompt(sample)
    # if "[[fewshot_examples]]" in full_prompt:
    #     print("✓ Full prompt contains [[fewshot_examples]] placeholder")
    #     full_examples = task.populate_full_examples(num_examples=5)
    #     final_prompt = full_prompt.replace("[[fewshot_examples]]", full_examples)
    #     print(f"✓ Replaced placeholder successfully")
    #     print(f"  Before: {len(full_prompt)} chars")
    #     print(f"  After:  {len(final_prompt)} chars")
    #     print(f"  Added:  {len(final_prompt) - len(full_prompt)} chars")
    # else:
    #     print("✗ Full prompt does not contain [[fewshot_examples]] placeholder")
    
    # # Test with concat prompt
    # concat_prompt = task.populate_concat_prompt(sample)
    # if "[[fewshot_examples]]" in concat_prompt:
    #     print("\n✓ Concat prompt contains [[fewshot_examples]] placeholder")
    #     sharded_examples = task.populate_sharded_examples(num_examples=5)
    #     final_prompt = concat_prompt.replace("[[fewshot_examples]]", sharded_examples)
    #     print(f"✓ Replaced placeholder successfully")
    #     print(f"  Before: {len(concat_prompt)} chars")
    #     print(f"  After:  {len(final_prompt)} chars")
    #     print(f"  Added:  {len(final_prompt) - len(concat_prompt)} chars")
    # else:
    #     print("✗ Concat prompt does not contain [[fewshot_examples]] placeholder")
    
    # print("\n" + "=" * 80)
    # print("TESTING COMPLETE")
    # print("=" * 80)