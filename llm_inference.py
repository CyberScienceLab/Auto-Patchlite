import ollama

def call_llm_ollama(prompt: str, model: str = 'mistral') -> str:
    """Call the Ollama LLM (e.g., mistral) and return the response text."""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
