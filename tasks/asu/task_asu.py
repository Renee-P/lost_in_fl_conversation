from typing import List, Dict, Any
from task_base import Task
import json, random, re 
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sacrebleu.metrics import CHRF
import gc    
import time

class TaskAS(Task):
    def __init__(self):
        with open("prompts/asu/asu_full_prompt.txt", "r") as f:
            self.fully_specified_prompt = f.read()
        with open("prompts/asu/asu_system_prompt.txt", "r") as f:
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

        self.answer_extraction_strategy = "full_response"

    def get_task_name(self) -> str:
        return "asu"

    def get_dataset_file(self) -> str:
        return "data/sharded_instructions.json"

    def get_samples(self, filter="full") -> List[Dict[str, Any]]:
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        return data

    def get_answer_description(self) -> str:
        return "The answer should be a one or two sentence summary."

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        """
        Evaluate if the extracted answer is correct using an ensemble of metrics.
        Returns average of BERTScore, ChrF++, and ROUGE-L F1.
        """
        
        # Get the reference summary from the sample
        gold = sample["label"].strip()

        # Determine device (use GPU if available, otherwise CPU)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        
        try:
            # Remove "BUOD: " prefix if present (case-insensitive)
            cleaned_answer = re.sub(r'^BUOD:\s*', '', extracted_answer, flags=re.IGNORECASE).strip()
            
            # Check if answer is empty after cleaning
            if not cleaned_answer:
                return {
                    "score": 0.0,
                    "bertscore": 0.0,
                    "chrf": 0.0,
                    "rouge_l": 0.0,
                    "error": f"Empty answer after cleaning: {repr(extracted_answer)}"
                }
            
            # 1. ROUGE-L F1
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
            rouge_scores = scorer.score(gold, cleaned_answer)
            rouge_l_f1 = rouge_scores['rougeL'].fmeasure
            
            # # 2. BERTScore (F1)
            # # Returns (P, R, F1) tensors
            # _, _, bert_f1 = bert_score_fn(
            #     [cleaned_answer],
            #     [gold],
            #     lang="other",  # Use "other" for Filipino
            #     device=device,
            #     rescale_with_baseline=False,
            #     verbose=False
            # )
            # bertscore = bert_f1.item()  # Convert tensor to float

            # 2. BERTScore (F1) with aggressive retry - MUST succeed
            bertscore = None
            max_retries = 100  # Keep trying until success!

            for attempt in range(max_retries):
                try:
                    # Force cleanup before each attempt
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Try to compute BERTScore
                    _, _, bert_f1 = bert_score_fn(
                        [cleaned_answer],
                        [gold],
                        lang="other",
                        device=device,
                        rescale_with_baseline=False,
                        verbose=False
                    )
                    bertscore = bert_f1.item()
                    
                    # Success! Break out of retry loop
                    if attempt > 0:
                        print(f"✅ BERTScore succeeded on attempt {attempt+1}")
                    break
                    
                except Exception as e:
                    error_str = str(e).lower()
                    is_memory_error = ("meta tensor" in error_str or 
                                    "memory" in error_str or
                                    "oom" in error_str)
                    
                    if is_memory_error:
                        # Memory error - will keep retrying
                        wait_time = min(2 ** min(attempt, 5), 16)  # 1s, 2s, 4s, 8s, 16s max
                        print(f"⚠️  BERTScore memory error (attempt {attempt+1}/{max_retries}), "
                            f"waiting {wait_time}s...")
                        
                        # Aggressive cleanup
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Wait for memory to free up
                        time.sleep(wait_time)
                    else:
                        # Non-memory error - can't retry this
                        print(f"❌ BERTScore error (not memory-related): {e}")
                        raise  # Re-raise to outer try-except

            # If we get here after all retries, something is very wrong
            if bertscore is None:
                raise RuntimeError(f"BERTScore failed after {max_retries} retry attempts. "
                                f"This should not happen - please check your system.")

            
            # 3. ChrF++
            chrf = CHRF(word_order=2)  # ChrF++ uses word order=2
            chrf_score = chrf.sentence_score(cleaned_answer, [gold]).score / 100.0  # Normalize to 0-1
            
            # Calculate ensemble average
            average_score = (rouge_l_f1 + bertscore + chrf_score) / 3.0
            # print(f"DETAILED SCORE: {chrf_score}, {rouge_l_f1}, {chrf_score}")
            
            return {
                "score": average_score,
                "rouge_l": rouge_l_f1,
                "bertscore": bertscore,
                "chrf": chrf_score,
            }
        
        except Exception as e:
            return {
                "score": 0.0,
                "rouge_l": 0.0,
                "bertscore": 0.0,
                "chrf": 0.0,
                "error": f"Error computing metrics: {repr(e)}"
            }

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return (
            self.fully_specified_prompt
            .replace("[[text]]", sample["text"])
        )

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Concatenate shards and prepend the system prompt.
        [[fewshot_examples]] is ignored here (handled elsewhere).
        """
        concatenated_shards = "\n".join(shard["shard"] for shard in sample["shards"])
        return f"{self.system_prompt.strip()}\n\n{concatenated_shards.strip()}"

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
        "task_id": "sharded-asu/24",
        "text": "Inirerekomenda na ang punong-tanggapan ng Bahay Woodhill sa Aberdeen at Bahay Gordon sa Inverurie na ilipat sa Liwasang Harlaw sa Inverurie. Ang layunin ay tungo sa mas maliit, mas nababagay, at mas murang mga opisina. Ang Aberdeenshire ay ang tanging lokal na awtoridad sa Scotland na mayroong punong-tanggapan sa labas ng hangganan nito. Sasabihin sa mga konsehal na ang pagpapanatili ng mga kasalukuyang opisina sa Aberdeen at Inverurie ay mangangailangan ng malaking pamumuhunan, at magpapahintulot ang konseho na may mas maraming espasyo sa opisina kaysa sa kinakailangan.",
        "label": "Inirerekomenda ang paglilipat ng mga punong-tanggapan ng Bahay Woodhill at Bahay Gordon sa Liwasang Harlaw.",
        "shards": [
            { 
                "shard_id": 1,
                "shard": "Kailangan ko ng buod ng isang artikulong Filipino. Ibabahagi ko ang mga bahagi ng artikulo ng paunti-unti habang natatanggap ko ang mga ito, inkosindera ang lahat ng mga bahagi. Unang bahagi: Inirerekomenda na ang punong-tanggapan ng Bahay Woodhill sa Aberdeen at Bahay Gordon sa Inverurie na ilipat sa Liwasang Harlaw sa Inverurie."
            },
            {
                "shard_id": 2,
                "shard": "Nakatanggap ako ng bago. Isa pang bahagi: Ang layunin ay tungo sa mas maliit, mas nababagay, at mas murang mga opisina."
            },
            {
                "shard_id": 3,
                "shard": "Mayroon na namang bagong ibabahagi. Isa pang bahagi: Ang Aberdeenshire ay ang tanging lokal na awtoridad sa Scotland na mayroong punong-tanggapan sa labas ng hangganan nito."
            },
            {
                "shard_id": 4,
                "shard": "Mayroon pa akong natanggap. Isa pang bahagi: Sasabihin sa mga konsehal na ang pagpapanatili ng mga kasalukuyang opisina sa Aberdeen at Inverurie ay mangangailangan ng malaking pamumuhunan, at magpapahintulot ang konseho na may mas maraming espasyo sa opisina kaysa sa kinakailangan."
            }
        ],
        "task": "asu"
    }

    task = TaskAS()
    
    # print("=" * 60)
    # print("TESTING TaskAS")
    # print("=" * 60)
    
    # # Test prompts
    # print("\nFULL PROMPT:")
    # print("-" * 60)
    # print(task.populate_fully_specific_prompt(sample))
    
    # print("\n\nCONCAT PROMPT:")
    # print("-" * 60)
    # print(task.populate_concat_prompt(sample))
    
    # Test evaluator function
    print("\n\n" + "=" * 60)
    print("TESTING EVALUATOR")
    print("=" * 60)
    
    # Test case 1: Perfect match
    test_answer_1 = "Inirerekomenda ang paglilipat ng mga punong-tanggapan ng Bahay Woodhill at Bahay Gordon sa Liwasang Harlaw."
    result_1 = task.evaluator_function(test_answer_1, sample)
    print(f"\nTest 1 - Perfect match:")
    print(f"Answer: {test_answer_1}")
    print(f"Result: {result_1}")
    
    # Test case 2: With "BUOD:" prefix
    test_answer_2 = "BUOD: Inirerekomenda ang paglilipat ng mga punong-tanggapan ng Bahay Woodhill at Bahay Gordon sa Liwasang Harlaw."
    result_2 = task.evaluator_function(test_answer_2, sample)
    print(f"\nTest 2 - With 'BUOD:' prefix:")
    print(f"Answer: {test_answer_2}")
    print(f"Result: {result_2}")
    
    # Test case 3: Partial match
    test_answer_3 = "Ang paglilipat ng punong-tanggapan ay inirerekomenda."
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
    test_answer_5 = "BUOD: "
    result_5 = task.evaluator_function(test_answer_5, sample)
    print(f"\nTest 5 - Only 'BUOD:' prefix:")
    print(f"Answer: '{test_answer_5}'")
    print(f"Result: {result_5}")
    
    print("\n" + "=" * 60)

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
