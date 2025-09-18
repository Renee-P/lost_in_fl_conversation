from ollama import Client
import json, re, time

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert isinstance(v, str), f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if keys_still_in_prompt:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages


class OllamaModel:
    def __init__(self, host="http://localhost:11434"):
        self.client = Client(host=host)

    def generate(self, messages, model="sailor2:1b", max_retries=3, temperature=0.7, variables={}):
        N = 0
        messages = format_messages(messages, variables)

        while True:
            try:
                response = self.client.chat(model=model, messages=messages, options={"temperature": temperature})
                response_text = response["message"]["content"]
                return response_text
            except Exception as e:
                N += 1
                if N >= max_retries:
                    raise e
                time.sleep(2)

    def generate_json(self, messages, model="sailor2:1b", **kwargs):
        response_text = self.generate(messages, model=model, **kwargs)
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Model did not return valid JSON: {response_text}")
        return parsed


# convenience functions
model = OllamaModel()
generate = model.generate
generate_json = model.generate_json


if __name__ == "__main__":
    # messages = [
    #     {"role": "user", "content": "Tell me a joke about UC Berkeley in JSON: {\"joke\": \"...\"}"}
    # ]

    # print(generate(messages, model="sailor2:1b"))
    # print(generate_json(messages, model="sailor2:1b"))

    messages = [
      {"role": "user", "content": "Humor is a way to make people laugh. Tell me a joke about AI."},
      {"role": "assistant", "content": '{"joke": '}
    ]

    model = "sailor2:1b"
    response = generate(messages, model=model)
      
    print(response)
