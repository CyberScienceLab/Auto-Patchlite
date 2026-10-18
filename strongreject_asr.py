# Wrapper for StrongREJECT string-matching ASR classifier
from strong_reject.evaluate import string_matching

def is_harmful_response_strongreject(prompt: str, response: str) -> bool:
    """Return True if the response is harmful (ASR=1.0), False if safe (ASR=0.0)."""
    result = string_matching(prompt, response)
    return result.get("score", 1.0) == 1.0
