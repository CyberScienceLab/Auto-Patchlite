def load_prompts_from_phi3_jsonl(file_path):
    """Load prompts and metadata from a phi3_allprompts.report.jsonl file."""
    import json
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Only process lines that are attempts (not eval, digest, etc)
                if entry.get("entry_type") == "attempt":
                    prompt_text = entry.get("prompt", {}).get("turns", [{}])[0].get("content", {}).get("text", "")
                    meta = {
                        "goal": entry.get("goal"),
                        "probe": entry.get("probe_classname"),
                        "detector": entry.get("detector_results"),
                        "uuid": entry.get("uuid"),
                        "seq": entry.get("seq"),
                        "status": entry.get("status"),
                    }
                    if prompt_text:
                        prompts.append({"prompt": prompt_text, "meta": meta})
            except Exception as e:
                print(f"⚠️ Error parsing phi3_allprompts line: {e}")
    return prompts
