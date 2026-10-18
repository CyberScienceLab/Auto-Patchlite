#!/usr/bin/env python3
"""
Add J1-J2 baselines to the strong baseline comparison table.

This script reuses the evaluation harness from strong_baseline_experiment.py
to ensure J1-J2 results are fully comparable to existing baselines.

Simply imports functions and re-runs evaluation with J1-J2 functions added.
"""

import sys
import json
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from strong_baseline_experiment
from strong_baseline_experiment import (
    load_attack_prompts,
    load_benign_prompts,
    evaluate_baseline,
)

# Import TDSC risk score for J1-A tuning
def prompt_risk_score(prompt: str) -> int:
    """Score prompt risk level (from TDSC)"""
    if not prompt:
        return 0
    t = prompt.lower()
    keywords = [
        "hack", "malware", "ransomware", "bomb", "shoot", "kill", "genocide",
        "harass", "xenophobic", "extort", "insider trading", "counterfeit", "sudo",
        "jailbreak", "bypass", "exploit", "weapon", "poison", "torture",
    ]
    score = sum(1 for word in keywords if word in t)
    if len(t) > 220:
        score += 1
    if "step-by-step" in t or "instructions" in t:
        score += 1
    return score

def deny_only_block(prompt: str, threshold: int = 2) -> bool:
    """
    J1-A: Deny-only baseline with threshold tuning.
    Blocks prompts with risk score >= threshold.
    """
    return prompt_risk_score(prompt) >= threshold

def strong_manual_baseline_block(prompt: str) -> bool:
    """
    J1-B / J2: Strong manual baseline (same as existing).
    Reused from strong_baseline_experiment.py.
    """
    prompt_lower = prompt.lower()
    
    attack_keywords = [
        'ignore all', 'ignore previous', 'forget all',
        'override', 'bypass', 'jailbreak',
        'system:', 'system prompt', 'new instructions',
        'pretend', 'act as', 'roleplay as',
        'you are now', 'from now on',
    ]
    
    for keyword in attack_keywords:
        if keyword in prompt_lower:
            return True
    
    if prompt_lower.count('instruction') > 2:
        return True
    
    if 'base64' in prompt_lower or 'base2048' in prompt_lower:
        return True
    
    if any(pattern in prompt_lower for pattern in ['<tool>', '</tool>', 'function_call', 'api_call']):
        return True
    
    retrieval_attack_patterns = [
        'search:', 'query:', 'retrieve:', 'lookup:',
        'database:', 'sql', 'select from', 'where 1=1',
        'file://', 'http://', 'https://',
        'eval(', 'exec(', '__import__',
    ]
    for pattern in retrieval_attack_patterns:
        if pattern in prompt_lower:
            return True
    
    if len(prompt) > 2000:
        return True
    
    return False

def main():
    print("="*80)
    print("J1-J2 Baselines: Extended Evaluation")
    print("="*80)
    
    # Load existing baseline results (in parent results/ directory)
    results_dir = Path(__file__).parent.parent / "results"
    existing_path = results_dir / "table_b1_baseline_comparison.csv"
    
    with existing_path.open() as f:
        existing_rows = list(csv.DictReader(f))

    def _row_benign_pass_rename(row: dict) -> dict:
        r = dict(row)
        if r.get("Benign_Pass_Rate", "").strip() == "" and r.get("Correctness_Rate"):
            r["Benign_Pass_Rate"] = r["Correctness_Rate"]
        r.pop("Correctness_Rate", None)
        return r

    existing_rows = [_row_benign_pass_rename(r) for r in existing_rows]
    
    print(f"\n[Step 1] Loaded existing baseline table: {len(existing_rows)} rows")
    
    # Load evaluation data (uses strong_baseline infrastructure)
    print("\n[Step 2] Loading evaluation data (using strong_baseline harness)...")
    harmful_prompts = load_attack_prompts()
    benign_prompts = load_benign_prompts()
    print(f"  Harmful: {len(harmful_prompts)}")
    print(f"  Benign: {len(benign_prompts)}")
    
    # Evaluate J1-A: Deny-only with threshold tuning
    print("\n[Step 3] Tuning J1-A deny-only baseline...")
    
    # Parse existing AutoPatch-Lite refusal rate to match
    autopatch_row = existing_rows[3]  # Row 4 is AutoPatch-Lite
    autopatch_refused_str = autopatch_row['Refused']  # Format: "132 (88.0%)"
    import re
    m = re.search(r'\((\d+(?:\.\d+)?)%\)', autopatch_refused_str)
    target_refusal_rate = float(m.group(1)) / 100.0 if m else 0.88
    
    print(f"  Target refusal rate: {target_refusal_rate:.1%}")
    
    best_threshold = 2
    best_diff = float('inf')
    
    for threshold in range(0, 8):
        results = evaluate_baseline(
            f"deny_threshold_{threshold}",
            harmful_prompts,
            benign_prompts,
            lambda p, t=threshold: deny_only_block(p, t)
        )
        
        refused_pct = results['Refused_Pct'] / 100.0
        diff = abs(refused_pct - target_refusal_rate)
        
        print(f"    Threshold {threshold}: refusal={refused_pct:.1%}, diff={diff:.4f}")
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
        
        if diff < 0.02:
            break
    
    print(f"  ✓ Selected threshold={best_threshold}")
    
    # Evaluate all three baselines
    results_new = []
    
    # Copy existing baselines to new table
    for row in existing_rows:
        results_new.append(row)
    
    # J1-A: Deny-only
    print("\n[Step 4] Evaluating baselines...")
    j1a_result = evaluate_baseline(
        'Deny-only (matched refusal)',
        harmful_prompts,
        benign_prompts,
        lambda p: deny_only_block(p, best_threshold)
    )
    
    results_new.append({
        'Baseline': j1a_result['baseline'],
        'ASR': f"{j1a_result['ASR']:.3f}",
        'Safety_Pass_Rate': f"{j1a_result['Safety_Pass_Rate']:.3f}",
        'N_Harmful': j1a_result['N_Harmful'],
        'N_Benign': j1a_result['N_Benign'],
        'Unchanged': f"{j1a_result['Unchanged']} ({j1a_result['Unchanged_Pct']:.1f}%)",
        'Changed_Correct': f"{j1a_result['Changed_Correct']} ({j1a_result['Changed_Correct_Pct']:.1f}%)",
        'Changed_Wrong': f"{j1a_result['Changed_Wrong']} ({j1a_result['Changed_Wrong_Pct']:.1f}%)",
        'Refused': f"{j1a_result['Refused']} ({j1a_result['Refused_Pct']:.1f}%)",
        'Benign_Pass_Rate': f"{j1a_result['Correctness_Rate']:.1f}%",
        'Avg_Latency_ms': f"{j1a_result['Avg_Latency_ms']:.2f}",
        'P95_Latency_ms': f"{j1a_result['P95_Latency_ms']:.2f}",
    })
    
    # J1-B: Sanitize/validate-only
    j1b_result = evaluate_baseline(
        'Sanitize/validate-only',
        harmful_prompts,
        benign_prompts,
        strong_manual_baseline_block
    )
    
    results_new.append({
        'Baseline': j1b_result['baseline'],
        'ASR': f"{j1b_result['ASR']:.3f}",
        'Safety_Pass_Rate': f"{j1b_result['Safety_Pass_Rate']:.3f}",
        'N_Harmful': j1b_result['N_Harmful'],
        'N_Benign': j1b_result['N_Benign'],
        'Unchanged': f"{j1b_result['Unchanged']} ({j1b_result['Unchanged_Pct']:.1f}%)",
        'Changed_Correct': f"{j1b_result['Changed_Correct']} ({j1b_result['Changed_Correct_Pct']:.1f}%)",
        'Changed_Wrong': f"{j1b_result['Changed_Wrong']} ({j1b_result['Changed_Wrong_Pct']:.1f}%)",
        'Refused': f"{j1b_result['Refused']} ({j1b_result['Refused_Pct']:.1f}%)",
        'Benign_Pass_Rate': f"{j1b_result['Correctness_Rate']:.1f}%",
        'Avg_Latency_ms': f"{j1b_result['Avg_Latency_ms']:.2f}",
        'P95_Latency_ms': f"{j1b_result['P95_Latency_ms']:.2f}",
    })
    
    # J2: Governance-only (same blocking as J1-B)
    j2_result = evaluate_baseline(
        'Strong Baseline + canary/rollback (no synthesis)',
        harmful_prompts,
        benign_prompts,
        strong_manual_baseline_block
    )
    
    results_new.append({
        'Baseline': j2_result['baseline'],
        'ASR': f"{j2_result['ASR']:.3f}",
        'Safety_Pass_Rate': f"{j2_result['Safety_Pass_Rate']:.3f}",
        'N_Harmful': j2_result['N_Harmful'],
        'N_Benign': j2_result['N_Benign'],
        'Unchanged': f"{j2_result['Unchanged']} ({j2_result['Unchanged_Pct']:.1f}%)",
        'Changed_Correct': f"{j2_result['Changed_Correct']} ({j2_result['Changed_Correct_Pct']:.1f}%)",
        'Changed_Wrong': f"{j2_result['Changed_Wrong']} ({j2_result['Changed_Wrong_Pct']:.1f}%)",
        'Refused': f"{j2_result['Refused']} ({j2_result['Refused_Pct']:.1f}%)",
        'Benign_Pass_Rate': f"{j2_result['Correctness_Rate']:.1f}%",
        'Avg_Latency_ms': f"{j2_result['Avg_Latency_ms']:.2f}",
        'P95_Latency_ms': f"{j2_result['P95_Latency_ms']:.2f}",
    })
    
    # Write extended results
    print("\n[Step 5] Writing results...")
    output_path = results_dir / "table_b1_extended_with_j1_j2.csv"
    
    with output_path.open('w', newline='') as f:
        fieldnames = list(results_new[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_new)
    
    print(f"  ✓ {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nExtended Baseline Table ({len(results_new)} rows):")
    for row in results_new:
        print(f"  {row['Baseline']:45s} ASR={row['ASR']:>7s}  Refused={row['Refused']:>15s}")
    
    print("\n" + "="*80)
    print("✅ J1-J2 Evaluation Complete")
    print(f"   Output: {output_path}")
    print("="*80)

if __name__ == '__main__':
    main()
