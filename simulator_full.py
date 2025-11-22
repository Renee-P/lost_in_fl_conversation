import random

from utils_log import log_conversation
from system_agent import SystemAgent
from model_genai import generate
from tasks import get_task
from utils import date_str
import copy # not in orig

# TO DO: add parameter is_base_model and if True will add 5-shot examples
# note: removed return_metadata and related (e.g. assistant_response_obj) for now - using free gemini api for sharding

class ConversationSimulatorFull:
    def __init__(self, sample, assistant_model, system_model, is_base_model=False, run_concat=False, 
                 run_shuffle_concat=False, temperature=1.0, dataset_fn=None, log_folder=None):
        self.task_name = sample["task"]
        self.task = get_task(self.task_name)
        # print("Active extraction strategy:", self.task.answer_extraction_strategy)
        self.dataset_fn = dataset_fn
        self.sample = sample
        self.assistant_model = assistant_model
        self.system_model = system_model
        self.is_base_model = is_base_model
        self.run_concat = run_concat
        self.run_shuffle_concat = run_shuffle_concat
        self.log_folder = log_folder
        self.run_custom_temperature = temperature != 1.0
        self.temperature = temperature
        
        self.system_agent = SystemAgent(self.task_name, self.system_model, self.sample)

    def run(self, verbose=False, save_log=True):  
        if self.run_shuffle_concat and self.run_concat:
            raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

        if self.run_shuffle_concat:
            conv_type = "shuffle-concat"

            sample_to_use = copy.deepcopy(self.sample)  # Copy before shuffle

            # Keep the first shard fixed, shuffle the rest
            first_shard = sample_to_use["shards"][0]
            rest_shards = sample_to_use["shards"][1:]
            random.shuffle([1,2,3])
            random.shuffle(rest_shards)

            # Combine first shard with shuffled rest
            sample_to_use["shards"] = [first_shard] + rest_shards

            input_prompt = self.task.populate_concat_prompt(sample_to_use)

            # random.shuffle(self.sample["shards"])

            # input_prompt = self.task.populate_concat_prompt(self.sample)
        elif self.run_concat:
            conv_type = "concat"
            # input_prompt = self.task.populate_concat_prompt(self.sample)

            sample_to_use = copy.deepcopy(self.sample)  # ← Create copy
            sample_to_use["shards"] = sorted(sample_to_use["shards"], key=lambda x: int(x["shard_id"]))  # ← Sort shards
            input_prompt = self.task.populate_concat_prompt(sample_to_use)
        else:
            conv_type = "full"
            input_prompt = self.task.populate_fully_specific_prompt(self.sample)

        # Fill 5-shot examples
        if self.is_base_model:
            if self.run_shuffle_concat or self.run_concat:
                fewshot_examples = self.task.populate_sharded_examples(num_examples=5)
            else:
                fewshot_examples = self.task.populate_full_examples(num_examples=5)
            
            # Replace placeholder in prompt
            input_prompt = input_prompt.replace("[[fewshot_examples]]", fewshot_examples)
        else:
            # Remove placeholder for instruction-tuned models
            input_prompt = input_prompt.replace("[[fewshot_examples]]", "")

        if verbose:                                     
            print(f"\033[92m[system] {input_prompt}\033[0m")

        # custom output dir for different temperatures to not mix up
        if self.run_custom_temperature:
            conv_type = f"{conv_type}-t{self.temperature}"

        is_reasoning_model = "o1" in self.assistant_model or "o3" in self.assistant_model or "deepseek-r1" in self.assistant_model or "gemini-2.5" in self.assistant_model
      
        max_tokens = 16000 if is_reasoning_model else 1000
        assistant_response = generate([{"role": "user", "content": input_prompt}], model=self.assistant_model, temperature=self.temperature, max_tokens=max_tokens)
        if verbose:
            print(f"\033[91m[assistant] {assistant_response}\033[0m")
        
        trace = [{"role": "user", "content": input_prompt}, {"role": "assistant", "content": assistant_response, "cost_usd": 0.0}]

        extracted_answer = self.system_agent.extract_answer(trace)
        # print("DEBUG: Extracted answer from system_agent.extract_answer():\n", repr(extracted_answer))

        evaluation_return = self.task.evaluator_function(extracted_answer, self.sample)
        # print("DEBUG: Evaluator result:", evaluation_return)
        assert type(evaluation_return) is dict and ("score" in evaluation_return or "is_correct" in evaluation_return), "Evaluator function should return a dictionary with 'score' or 'is_correct' key"
        score = evaluation_return.get("score", None)
        is_correct = score == 1.0

        trace.append({"role": "log", "content": {"type": "answer-evaluation", "exact_answer": extracted_answer, "is_correct": is_correct, "score": score, "evaluation_return": evaluation_return}, "timestamp": date_str()})

        if verbose:
            print('==================================================')
            icon = "\033[92m✔\033[0m" if is_correct else "\033[91m✘\033[0m"
            print(f"{icon} {extracted_answer} (score: {score})")

        if save_log:
            log_conversation(conv_type, self.task_name, self.sample["task_id"], self.dataset_fn, assistant_model=self.assistant_model, system_model="NA", user_model="NA", trace=trace, is_correct=is_correct, score=score, log_folder=self.log_folder)
        return is_correct, score

if __name__ == "__main__":
    import json, argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--system_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--task", type=str, default="qa")
    parser.add_argument("--is_base_model", action="store_true")
    parser.add_argument("--run_concat", action="store_true")
    parser.add_argument("--run_shuffle_concat", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if args.run_concat and args.run_shuffle_concat:
        raise ValueError("Cannot set both run_concat and run_shuffle_concat to True")

    with open("data/sharded_mt.json", "r", encoding='utf-8') as f:
        data = json.load(f)

    # data = [d for d in data if (d["task"] == args.task or args.task == "all")]
    data = [d for d in data if (d["task"] == args.task and d["task_id"] == "sharded-mt/73t")]

    sample = random.choice(data)

    conversation_simulator = ConversationSimulatorFull(
        sample, 
        args.assistant_model, 
        args.system_model, 
        is_base_model=args.is_base_model,
        run_concat=args.run_concat, 
        run_shuffle_concat=args.run_shuffle_concat,
        temperature=args.temperature,
        dataset_fn="data/sharded_mt.json",  
        log_folder="logs"  # ADD THIS
    )
    
    for _ in range(1):
        conversation_simulator.run(verbose=args.verbose, save_log=True)