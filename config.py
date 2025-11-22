import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

config = {
    "base_dir": BASE_DIR,
    "data_dir": os.path.join(BASE_DIR, "data"),
    "raw_data_dir": os.path.join(BASE_DIR, "data", "raw"),
    "final_data_dir": os.path.join(BASE_DIR, "data"),
    "prompts_dir": os.path.join(BASE_DIR, "prompts"),

    "tasks": [
        "asu",
        "cr",
        "nli",
        "pi",
        "qa",
        "sa",
        "td",
        "mt"
    ],

    "raw_file_paths": {
        "asu": os.path.join("asu", "tl_xlsum.jsonl"),
        "cr": os.path.join("cr", "tl_balanced_copa.jsonl"),
        "nli": os.path.join("nli", "tl_xnli.jsonl"),
        "pi": os.path.join("pi", "tl_paws.jsonl"),
        "qa": os.path.join("qa", "tl_belebele.jsonl"),
        "sa": os.path.join("sa", "tl_elections_sentiment.jsonl"),
        "td": os.path.join("td", "tl_elections_hsd.jsonl"),
        "mt": {
            "en_to_tgl": "mt/en_to_tgl_Latn.jsonl",
            "tgl_to_en": "mt/tgl_Latn_to_en.jsonl",
            "en_to_tgl_examples": "mt/en_to_tgl_Latn_examples.jsonl",
            "tgl_to_en_examples": "mt/tgl_Latn_to_en_examples.jsonl"
        }
    },

    "raw_file_names": {
        "asu": "tl_xlsum.jsonl",
        "cr": "tl_balanced_copa.jsonl",
        "nli": "tl_xnli.jsonl",
        "pi": "tl_paws.jsonl",
        "qa": "tl_belebele.jsonl",
        "sa": "tl_elections_sentiment.jsonl",
        "td": "tl_elections_hsd.jsonl"
    },

    # idt this is needed
    "segment_prompt_files": {
        "asu": "asu_segment.txt",
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
