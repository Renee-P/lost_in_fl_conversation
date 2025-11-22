import json
import os
import git
# import time
# import pandas as pd
from bson.objectid import ObjectId
# from collections import Counter
# from datetime import datetime

def get_log_files(conv_type, task_name, assistant_model, force_create=False, log_folder="logs"):
    # Sanitize the assistant_model name for Windows compatibility
    # Replace characters that are invalid in Windows filenames: < > : " / \ | ? *
    sanitized_model = assistant_model
    for char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
        sanitized_model = sanitized_model.replace(char, '_')

    base_log_file = f"{log_folder}/{task_name}/{conv_type}/{conv_type}_{task_name}_{sanitized_model}.jsonl"

    # if the folder doesn't exist, create it
    if not os.path.exists(os.path.dirname(base_log_file)):
        if not force_create:
            return []
        os.makedirs(os.path.dirname(base_log_file))

    # Get all matching log files including split files
    log_dir = os.path.dirname(base_log_file)
    base_name = os.path.basename(base_log_file).replace(".jsonl", "")
    log_files = []

    for file in os.listdir(log_dir):
        if file.startswith(base_name) and file.endswith(".jsonl"):
            log_files.append(os.path.join(log_dir, file))

    # if it doesn't exist, touch it
    if len(log_files) == 0:
        if force_create:
            with open(base_log_file, "w") as f:
                f.write("")
            log_files.append(base_log_file)
        else:
            return []

    return sorted(log_files)  # Sort to ensure consistent order

def log_conversation(conv_type, task_name, task_id, dataset_fn, assistant_model, system_model, user_model, trace, is_correct=None, score=None, additional_info={}, log_folder=None):
    log_files = get_log_files(conv_type, task_name, assistant_model, force_create=True, log_folder=log_folder)
    log_file = log_files[-1]

    if dataset_fn:
        dataset_fn = dataset_fn.split("/")[-1]
    else:
        dataset_fn = "unknown"

    # if the folders don't exist, create them
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    git_version = git.Repo(search_parent_directories=True).head.object.hexsha

    record = {"conv_id": str(ObjectId()), "conv_type": conv_type, "task": task_name, "task_id": task_id, "dataset_fn": dataset_fn, "assistant_model": assistant_model, "system_model": system_model, "user_model": user_model, "git_version": git_version, "trace": trace, "is_correct": is_correct, "score": score} # , "source_conv_id": source_conv_id
    record.update(additional_info) # sample-specific, for example for recap
    with open(log_file, "a") as f:
        f.write(json.dumps(record)+"\n")