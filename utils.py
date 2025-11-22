from datetime import datetime
import os

def print_colored(text, color):
    if color == "red":
        print(f"\033[91m{text}\033[0m")
    elif color == "green":
        print(f"\033[92m{text}\033[0m")
    elif color == "blue":
        print(f"\033[94m{text}\033[0m")
    elif color == "purple":
        print(f"\033[95m{text}\033[0m")
    else:
        raise Exception(f"Unknown color: {color}")

def extract_conversation(simulation_trace, to_str=False, skip_system=False, only_last_turn=False):
    keep_roles = ["system", "assistant", "user"] if not skip_system else ["assistant", "user"]
    real_conversation = [msg for msg in simulation_trace if msg["role"] in keep_roles]
    if only_last_turn:
        # get all the user turns
        user_turn_idxs = [i for i, msg in enumerate(real_conversation) if msg["role"] == "user"]
        last_user_turn_idx = user_turn_idxs[-1]
        real_conversation = real_conversation[last_user_turn_idx + 1:]

    real_conversation = [{"role": msg["role"], "content": msg["content"]} for msg in real_conversation] # only keep role and content

    if to_str:
        return "\n\n".join([f"[{msg['role']}] {msg['content']}" for msg in real_conversation])
    else:
        return real_conversation

def date_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")