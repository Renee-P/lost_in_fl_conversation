import os, json, time, re, random, copy
import threading
from google.api_core.exceptions import ResourceExhausted, InternalServerError, Aborted, DeadlineExceeded
from google.genai.errors import ClientError, ServerError  
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
import httpx

# Load .env
load_dotenv()

def format_messages(messages, variables={}):
    """
    Formats the messages list, substituting variables in the last user message.
    System messages are separated and returned.
    """
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

    # Separate system messages from the main conversation history - for genai api
    genai_messages = copy.deepcopy(messages)

    system_message = None
    for msg in genai_messages:
        if msg["role"] == "system":
            system_message = msg["content"]
            break
    
    # Filter out system messages from the conversation history passed to the API
    genai_messages = [{"role": "model" if msg["role"] == "assistant" else msg["role"], "parts": [{"text": msg["content"]}]} for msg in genai_messages if msg["role"] != "system"]

    return genai_messages, system_message

class GeminiModel:
    def __init__(self):
        keys_raw = os.getenv("GEMINI_API_KEYS", "")
        self.api_keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        if not self.api_keys:
            raise ValueError("No API keys found. Set GEMINI_API_KEYS in .env")

        self.current_key_index = 0
        self.client = self._new_client(self.api_keys[self.current_key_index])
        self.rotation_lock = threading.Lock()

    def _new_client(self, key):
        return genai.Client(api_key=key)

    def _rotate_key(self):
        with self.rotation_lock:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            self.client = self._new_client(self.api_keys[self.current_key_index])
            print(f"[rotation] Switched to key index {self.current_key_index}")

    def _handle_api_exception(self, e, attempt, max_retries):
        """
        Handles API-related exceptions and determines the next action (retry/wait/rotate/raise).
        Returns True if the process should continue (retry), False if it should stop (re-raise).
        """
        error_message = str(e)
        error_code = getattr(e, 'code', None)
        
        # 1. Handle Rate Limit (429 RESOURCE_EXHAUSTED)
        is_rate_limit = (
            isinstance(e, ResourceExhausted) or 
            (isinstance(e, ClientError) and error_code == 429) or
            "429" in error_message or 
            "RESOURCE_EXHAUSTED" in error_message
        )
        
        if is_rate_limit:
            print(f"[Rate Limit] Attempt {attempt}/{max_retries}: {error_message}")
            
            # Parse retry delay from error message
            retry_delay = 60
            match = re.search(r'retry in ([\d.]+)s', error_message)
            if match:
                retry_delay = int(float(match.group(1))) + 5
            
            if "per minute" in error_message.lower() or "perminute" in error_message.lower():
                print(f"[rate-limit] Per-minute limit. Waiting {retry_delay}s...")
                time.sleep(retry_delay)
                return True

            if "per day" in error_message.lower() or "perday" in error_message.lower():
                if len(self.api_keys) > 1:
                    print("[rate-limit] Daily limit hit. Rotating key...")
                    self._rotate_key()
                    return True
                else:
                    print("Daily limit reached and no backup keys available.")
                    return False
            
            print(f"[rate-limit] Waiting {retry_delay}s before retry...")
            time.sleep(retry_delay)
            return True
        
        # 2. Handle Server Errors (500, 503 UNAVAILABLE) - ADD THIS
        is_server_error = (
            isinstance(e, (InternalServerError, ServerError)) or
            error_code in [500, 503] or
            "503" in error_message or
            "500" in error_message or
            "UNAVAILABLE" in error_message or
            "overloaded" in error_message.lower()
        )
        
        if is_server_error:
            wait_time = min((2 ** attempt) * 10, 120)  # Exponential backoff, max 2 min
            print(f"[Server Error] Attempt {attempt}/{max_retries}: {error_message[:100]}")
            print(f"[Server Error] Model overloaded. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            return True
        
        # 3. Handle Network/Connection Errors (RemoteProtocolError, etc.) 
        is_network_error = (
            isinstance(e, httpx.RemoteProtocolError) or
            isinstance(e, httpx.ConnectError) or
            isinstance(e, httpx.TimeoutException) or
            "RemoteProtocolError" in error_message or
            "Server disconnected" in error_message or
            "Connection" in error_message
        )
        
        if is_network_error:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"[Network Error] Attempt {attempt}/{max_retries}. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            return True

        # 4. Handle Transient Errors (Aborted, DeadlineExceeded)
        elif isinstance(e, (Aborted, DeadlineExceeded)):
            wait_time = (2**attempt) + random.uniform(0, 1)
            print(f"[Transient Error] {e.__class__.__name__}. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            return True
            
        # 4. Non-retryable
        else:
            print(f"[Non-Retryable Error] Attempt {attempt+1}/{max_retries}: {e.__class__.__name__}")
            print(f"  Message: {error_message[:200]}")
            return False

    def generate(self, messages, model="gemini-2.5-flash", max_retries=10, temperature=1.0, is_json=False, response_schema=None, max_tokens=None, variables={}):
        """
        Generates content from the model with retry logic.
        Handles both text and JSON (via is_json/response_schema) generation.
        """
        # Format messages and extract system instructions
        genai_messages, system_message = format_messages(messages, variables)

        # Configure the request
        kwargs = {"temperature": temperature}
        if system_message:
            kwargs["system_instruction"] = system_message
        if is_json:
            kwargs["response_mime_type"] = "application/json"
        if response_schema:
            # Assuming response_schema is a pydantic model or dict/list structure (??)
            kwargs["response_schema"] = response_schema
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        # thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking

        attempt = 0
        while attempt < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=genai_messages,
                    config=GenerateContentConfig(**kwargs),
                )
                
                if is_json:
                    # If JSON generation, return the parsed object
                    return json.loads(response.text)
                else:
                    # Otherwise, return the raw text
                    return response.text

            except Exception as e:
                # If max retries is hit, or the handler determines it's non-recoverable, raise
                if attempt >= max_retries:
                    raise e

                attempt += 1
                # Delegate error handling and decision making to the separate function
                should_retry = self._handle_api_exception(e, attempt, max_retries)
                
                if not should_retry:
                    # If handler says "stop," re-raise the original exception
                    raise e 
                
        raise Exception(f"Failed to generate content from GenAI after {max_retries} attempts.")

    def generate_json(self, messages, model="gemini-2.5-flash", max_retries=10, temperature=1.0, is_json=True, response_schema=None, max_tokens=None, variables={}):
        """
        Convenience wrapper for generate() to enforce JSON output and parse the result.
        """
        return self.generate(
            messages=messages,
            model=model, 
            temperature=temperature, 
            max_retries=max_retries, 
            is_json=True, 
            response_schema=response_schema,
            max_tokens=max_tokens,
            variables=variables
        )
        
# convenience functions
model = GeminiModel()
generate = model.generate
generate_json = model.generate_json


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "Only reply in Tagalog."},
        {"role": "user", "content": "Humor is a way to make people laugh. Tell me a joke"},
        {"role": "assistant", "content": "joke: "},
    ]

    messages1 = [
        {"role": "user", "content": "Humor is a way to make people laugh."},
        {"role": "assistant", "content": "joke: "},
        {"role": "user", "content": "Tell me a joke in JSON with key 'joke'."}
    ]

    print(generate(messages))
    # print(messages)
    print(generate_json(messages1))

    # ------------------------------------------------------------------
    # --- VARIABLE TESTS ---
    # ------------------------------------------------------------------
    
    # sample_text = (
    #     "The sun is the star at the center of the Solar System. "
    #     "It is a nearly perfect sphere of hot plasma, heated to incandescence "
    #     "by nuclear fusion reactions in its core, radiating the energy mainly as light, "
    #     "ultraviolet, and infrared radiation. It is by far the most important "
    #     "source of energy for life on Earth."
    # )

    # messages_var_text = [
    #     {"role": "user", "content": "Please read the following text and provide a concise, one-sentence summary: [[text_to_summarize]]"},
    # ]

    # messages_var_json = [
    #     {"role": "user", "content": "Analyze the following text: [[text_to_summarize]]. Output a JSON object with two keys: 'topic' and 'summary'."}
    # ]

    # # Define the variable dictionary
    # variables_to_test = {"text_to_summarize": sample_text}
    
    # # Test 1: Standard Text Response with Variable
    # print("\n--- Testing Text Response with Variables ---")
    # response_text = generate(
    #     messages=messages_var_text,
    #     variables=variables_to_test
    # )
    # print(f"Text Response: {response_text}")

    # # Test 2: JSON Response with Variable
    # print("\n--- Testing JSON Response with Variables ---")
    # response_json = generate_json(
    #     messages=messages_var_json,
    #     variables=variables_to_test
    # )
    # print(f"JSON Response (parsed): {response_json}")

    # print("-----------------")
    # print(messages_var_text)
    # print("-----------------")
    # print(messages_var_json)