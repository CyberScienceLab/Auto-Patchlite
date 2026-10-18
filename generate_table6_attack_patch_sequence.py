import os
import json
import csv

# Table 6: Attack-Patch Sequence During Campaign (Per Round)
# Define the mapping for each round (customize as needed)
attack_rounds = [
    {"Rnd": 1, "Attack Family": "Jailbreak (direct)", "Benchmark Source": "Liu et al. 2024 / Awesome PI", "Variant": "v1", "Patch ID": "P101"},
    {"Rnd": 2, "Attack Family": "Indirect injection", "Benchmark Source": "RAG hidden-prompt document", "Variant": "v2", "Patch ID": "P205"},
    {"Rnd": 3, "Attack Family": "Tool misuse", "Benchmark Source": "OWASP (SQLi/XSS/OS-cmd)", "Variant": "v3", "Patch ID": "P150"},
    {"Rnd": 4, "Attack Family": "Jailbreak (obfuscated)", "Benchmark Source": "Liu et al. 2024 / Awesome PI", "Variant": "v4", "Patch ID": "P312"},
]

def get_metric_value(metrics, key, default):
    return metrics.get(key, default)

def main():
    table6_rows = []
    for round_info in attack_rounds:
        patch_id = round_info["Patch ID"]
        # Try to load the corresponding metrics file
        metrics_path = os.path.join("deployment_history", f"{patch_id}_metrics.json")
        if not os.path.exists(metrics_path):
            # Try campaign/ablation/baseline files as needed
            metrics_path = os.path.join("deployment_history", f"campaign_{patch_id}_metrics.json")
            if not os.path.exists(metrics_path):
                table6_rows.append([
                    round_info["Rnd"],
                    round_info["Attack Family"],
                    round_info["Benchmark Source"],
                    round_info["Variant"],
                    "XX.X", patch_id, "-YY.Y", "TT"
                ])
                continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        asr_before = get_metric_value(metrics, "asr_before", "XX.X")
        delta_asr = get_metric_value(metrics, "delta_asr", "-YY.Y")
        ttm = get_metric_value(metrics, "ttm", "TT")
        table6_rows.append([
            round_info["Rnd"],
            round_info["Attack Family"],
            round_info["Benchmark Source"],
            round_info["Variant"],
            asr_before,
            patch_id,
            delta_asr,
            ttm
        ])
    os.makedirs("table_results", exist_ok=True)
    with open("table_results/table6_attack_patch_sequence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rnd", "Attack Family", "Benchmark Source", "Variant", "ASR Before", "Patch ID", "ΔASR (pp)", "TTM (min)"])
        writer.writerows(table6_rows)
    print("Table 6 generated: table_results/table6_attack_patch_sequence.csv")

if __name__ == "__main__":
    main()
