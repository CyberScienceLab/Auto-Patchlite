#!/usr/bin/env python3
"""
Run Part B: Minimal Experiments

This script runs the three Part B experiments that address reviewer concerns:
- E1: Independent Safety Labeling Audit
- E2: Utility Decomposition on Benign Tasks
- E3: Strong Baseline Comparison
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
        'name': 'E1: Independent Safety Labeling Audit',
        'path': 'safety_labeling_audit/safety_labeling_audit_experiment.py',
        'description': 'Resolves circular safety labeling with independent judge evaluation'
    },
    {
        'name': 'E2: Utility Decomposition on Benign Tasks',
        'path': 'utility_decomposition/utility_decomposition_experiment.py',
        'description': 'Resolves utility contradiction by categorizing outcomes'
    },
    {
        'name': 'E3: Strong Baseline Comparison',
        'path': 'strong_baseline/strong_baseline_experiment.py',
        'description': 'Resolves weak-baselines criticism with expert-authored baseline'
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
    """Run all Part B experiments"""
    print("="*60)
    print("Part B: Minimal Experiments Runner")
    print("="*60)
    print("\nThese experiments address reviewer concerns:")
    print("  - E1: Independent safety labeling (circular labeling)")
    print("  - E2: Utility decomposition (utility contradiction)")
    print("  - E3: Strong baseline (weak-baselines criticism)")
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
    print("Part B Experiments Summary")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"  {status}: {result['name']}")
    
    print(f"\nTotal: {len(results)} experiments")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Output file locations
    print("\n" + "="*60)
    print("Output Files")
    print("="*60)
    print("\nE1 Results:")
    print("  - results/table_s1_confusion_matrix.csv")
    print("  - results/table_s1_per_family_metrics.csv")
    print("  - results/table_s1_overall_summary.csv")
    print("  - results/safety_labeling_audit_detailed.csv")
    print("  - results/safety_labeling_audit_summary.txt")
    
    print("\nE2 Results:")
    print("  - results/table_u1_utility_decomposition.csv")
    print("  - results/table_u1_overall_summary.csv")
    print("  - results/utility_decomposition_detailed.csv")
    print("  - results/utility_decomposition_summary.txt")
    
    print("\nE3 Results:")
    print("  - results/table_b1_baseline_comparison.csv")
    print("  - results/strong_baseline_detailed.csv")
    print("  - results/strong_baseline_summary.txt")
    
    if failed > 0:
        print("\n⚠ Some experiments failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("\n✓ All Part B experiments completed successfully!")
        sys.exit(0)

if __name__ == '__main__':
    main()

