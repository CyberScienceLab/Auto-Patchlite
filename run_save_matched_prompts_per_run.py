"""
Script: run_save_matched_prompts_per_run.py
After each run (baseline, ablation, campaign), copy matched_prompts.json to a uniquely named file in deployment_history/ for later per-prompt analysis.
"""
import os
import sys
from shutil import copyfile

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_save_matched_prompts_per_run.py <run_name>")
        print("Example: python run_save_matched_prompts_per_run.py baseline_no_defense")
        sys.exit(1)
    run_name = sys.argv[1]
    src = os.path.join('deployment_history', 'matched_prompts.json')
    dst = os.path.join('deployment_history', f'matched_prompts_{run_name}.json')
    if os.path.exists(src):
        copyfile(src, dst)
        print(f"[MatchedPrompts] Copied {src} to {dst}")
    else:
        print(f"[MatchedPrompts] Source file {src} does not exist.")

if __name__ == '__main__':
    main()
