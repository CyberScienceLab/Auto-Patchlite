#!/usr/bin/env python3
"""
Run all experiments in the new_experiments directory.
This script executes all experiments in the correct order.
"""

import os
import sys
import subprocess
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR

experiments = [
    {
        'name': 'CEPG Experiment',
        'path': 'CEPG/cepg_experiment.py',
        'description': 'Graph-based patch anomaly detection'
    },
    {
        'name': 'Survival Analysis',
        'path': 'survival_analysis/survival_experiment.py',
        'description': 'Patch survival under adaptive attacks'
    },
    {
        'name': 'Rollout Ablation',
        'path': 'rollout_ablation/rollout_experiment.py',
        'description': 'Canary vs immediate deployment comparison'
    },
    {
        'name': 'Code Vulnerability Benchmark',
        'path': 'code_vuln/code_vuln_experiment.py',
        'description': 'Juliet suite vulnerability patching'
    },
    {
        'name': 'Alignment Baseline',
        'path': 'alignment_baseline/alignment_experiment.py',
        'description': 'Alignment-only vs AutoPatch-Lite comparison'
    },
    {
        'name': 'Benign Quality Metrics',
        'path': 'benign_quality/benign_quality_experiment.py',
        'description': 'BLEU/ROUGE/LLM-judge evaluation'
    }
]

def run_experiment(exp_path: str, exp_name: str) -> bool:
    """Run a single experiment"""
    full_path = EXPERIMENTS_DIR / exp_path
    
    if not full_path.exists():
        print(f"ERROR: Experiment file not found: {full_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Path: {exp_path}")
    print(f"{'='*60}\n")
    
    # Change to experiment directory
    exp_dir = full_path.parent
    original_dir = os.getcwd()
    
    try:
        os.chdir(exp_dir)
        result = subprocess.run(
            [sys.executable, full_path.name],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ {exp_name} completed successfully")
            return True
        else:
            print(f"\n✗ {exp_name} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ {exp_name} failed with exception: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    """Run all experiments"""
    print("="*60)
    print("AutoPatch-Lite New Experiments Runner")
    print("="*60)
    print(f"\nRunning {len(experiments)} experiments...\n")
    
    results = []
    for exp in experiments:
        success = run_experiment(exp['path'], exp['name'])
        results.append({
            'name': exp['name'],
            'success': success
        })
    
    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"  {status}: {result['name']}")
    
    print(f"\nTotal: {len(results)} experiments")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\n⚠ Some experiments failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")
        sys.exit(0)

if __name__ == '__main__':
    main()

