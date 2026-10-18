"""
Script: run_repeat_experiment.py
Repeat a given experiment script multiple times (for statistical significance), saving metrics and matched prompts for each run with unique names.
"""
import os
import sys
import subprocess
from datetime import datetime

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_repeat_experiment.py <script.py> <num_runs> [run_name_prefix]")
        sys.exit(1)
    script = sys.argv[1]
    num_runs = int(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(script))[0]
    for i in range(num_runs):
        print(f"[Repeat] Running {script} (run {i+1}/{num_runs})...")
        subprocess.run([sys.executable, script])
        # Save matched prompts and metrics with unique names
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        run_name = f"{prefix}_run{i+1}_{timestamp}"
        # Save matched prompts
        src_mp = os.path.join('deployment_history', 'matched_prompts.json')
        dst_mp = os.path.join('deployment_history', f'matched_prompts_{run_name}.json')
        if os.path.exists(src_mp):
            from shutil import copyfile
            copyfile(src_mp, dst_mp)
            print(f"[Repeat] Copied {src_mp} to {dst_mp}")
        # Save metrics (if baseline/ablation/campaign script saves a metrics file, copy it)
        for fname in os.listdir('deployment_history'):
            if fname.endswith('_metrics.json') and not fname.startswith('live_metrics'):
                src = os.path.join('deployment_history', fname)
                dst = os.path.join('deployment_history', f"{run_name}_{fname}")
                if os.path.exists(src):
                    from shutil import copyfile
                    copyfile(src, dst)
                    print(f"[Repeat] Copied {src} to {dst}")

if __name__ == '__main__':
    main()
