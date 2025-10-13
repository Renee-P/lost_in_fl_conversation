import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    "base_dir": BASE_DIR,
    "data_dir": os.path.join(BASE_DIR, "data"),
    "raw_data_dir": os.path.join(BASE_DIR, "data", "raw"),
    "final_data_dir": os.path.join(BASE_DIR, "data", "final"),
    "prompts_dir": os.path.join(BASE_DIR, "prompts"),

    "tasks": [
        "as",
        "cr",
        "nli",
        "pi",
        "qa",
        "sa",
        "td"
    ],

    "raw_file_names": {
        "as": "tl_xlsum.jsonl",
        "cr": "tl_balanced_copa.jsonl",
        "nli": "tl_xnli.jsonl",
        "pi": "tl_paws.jsonl",
        "qa": "tl_belebele.jsonl",
        "sa": "tl_elections_sentiment.jsonl",
        "td": "tl_elections_hsd.jsonl"
    },

    # idt this is needed
    "segment_prompt_files": {
        "as": "as_segment.txt",
        "cr": "cr_segment.txt",
        "nli": "nli_segment.txt",
        "pi": "pi_segment.txt",
        "qa": "qa_segment.txt",
        "sa": "sa_segment.txt",
        "td": "td_segment.txt"
    },

    # do we need this?
    "model": "sailor2:1b",
    "ollama_host": "http://localhost:11434",
    "max_retries": 3,
    "temperature": 1.0,
}
