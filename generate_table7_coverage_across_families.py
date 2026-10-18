import os
import json
import csv

# Table 7: Coverage Across Attack Families During Campaign
# Define attack families and mapping to metrics files or patch types
attack_families = [
    {"Attack Family": "Jailbreak / Prompt Suite", "family_key": "jailbreak"},
    {"Attack Family": "Indirect Injection (RAG)", "family_key": "indirect_injection"},
    {"Attack Family": "Malicious Tool Use (OWASP-inspired)", "family_key": "tool_misuse"},
    {"Attack Family": "Retrieval Attack Data (hidden prompts)", "family_key": "retrieval_attack"},
    {"Attack Family": "Code Vulnerability Scenarios (ManyBugs/Juliet)", "family_key": "code_vuln"},
]

def get_family_metrics(family_key):
    # Try to find ablation or campaign metrics for this family
    ablation_path = os.path.join("deployment_history", f"ablation_{family_key}_metrics.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            return json.load(f)
    # Fallback: try campaign or other files as needed
    return None

def main():
    table7_rows = []
    N_total = 0
    M_total = 0
    for fam in attack_families:
        metrics = get_family_metrics(fam["family_key"])
        N = metrics.get("n_attempts", 0) if metrics else 0
        M = metrics.get("n_mitigated", 0) if metrics else 0
        coverage = 100 * M / N if N else 0
        residual_asr = metrics.get("residual_asr", "R") if metrics else "R"
        table7_rows.append([
            fam["Attack Family"],
            N,
            M,
            f"{coverage:.1f}",
            residual_asr
        ])
        N_total += N
        M_total += M
    overall_coverage = 100 * M_total / N_total if N_total else 0
    table7_rows.append([
        "Overall",
        N_total,
        M_total,
        f"{overall_coverage:.1f}",
        "RΣ"
    ])
    os.makedirs("table_results", exist_ok=True)
    with open("table_results/table7_coverage_across_families.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Attack Family", "N (#Attempts)", "# Mitigated", "Coverage (%)", "Residual ASR (%)"])
        writer.writerows(table7_rows)
    print("Table 7 generated: table_results/table7_coverage_across_families.csv")

if __name__ == "__main__":
    main()
