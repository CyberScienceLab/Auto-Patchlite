import os
import json
import csv


# Table 8: Canary Catch and Rollback Summary vs Immediate Rollout (No-Canary)
# Use patch IDs from Table 6
patch_ids = [
    "e797d27a",
    "e61ba3b3",
    "0cd819e5",
    "3801858c",
]



def get_canary_status_and_impact(patch_id):
    """
    Parse deployment_history/patch_{patch_id}_result.jsonl for canary/rollback status.
    Count users/prompts impacted from per_prompt_results.jsonl for canary and no-canary.
    Returns: (issue_detected, rollback, users_impacted_full, users_impacted_no_canary)
    """
    result_path = os.path.join("deployment_history", f"patch_{patch_id}_result.jsonl")
    per_prompt_path = os.path.join("deployment_history", f"per_prompt_results_{patch_id}.jsonl")
    issue_detected = "No"
    rollback = "No"
    users_impacted_full = 0
    users_impacted_no_canary = 0
    # Canary/rollback status
    if os.path.exists(result_path):
        print(f"[DEBUG] Reading deployment result: {result_path}")
        with open(result_path, "r") as f:
            lines = f.readlines()
        print(f"[DEBUG] Found {len(lines)} lines in {result_path}")
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            status = rec.get("status", "")
            phase = rec.get("phase", "")
            if status == "rolled_back" and phase == "canary":
                issue_detected = "Yes"
                rollback = "Yes"
            elif status in ("canary_passed", "deployed"):
                issue_detected = "No"
                rollback = "No"
    # Impacted prompts/users: count unique prompt texts as proxy
    # For canary: count prompts where patched==True (blocked)
    # For no-canary: count prompts where is_harmful==True and patched==False
    if os.path.exists(per_prompt_path):
        print(f"[DEBUG] Reading per-prompt results: {per_prompt_path}")
        prompts_blocked = set()
        prompts_harmful = set()
        total_lines = 0
        with open(per_prompt_path, "r") as f:
            for line in f:
                total_lines += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                prompt = rec.get("prompt")
                if prompt is None:
                    continue
                if rec.get("patched"):
                    prompts_blocked.add(prompt)
                if rec.get("is_harmful") and not rec.get("patched"):
                    prompts_harmful.add(prompt)
        print(f"[DEBUG] {patch_id}: {len(prompts_blocked)} blocked, {len(prompts_harmful)} harmful, {total_lines} total lines")
        users_impacted_full = len(prompts_blocked)
        users_impacted_no_canary = len(prompts_harmful)
    else:
        print(f"[DEBUG] Per-prompt file not found: {per_prompt_path}")
    return issue_detected, rollback, users_impacted_full, users_impacted_no_canary


def main():
    table8_rows = []
    total_blocked = 0
    total_harmful = 0
    for patch_id in patch_ids:
        print(f"\n[INFO] Processing patch {patch_id}")
        issue_detected, rollback, users_impacted_full, users_impacted_no_canary = get_canary_status_and_impact(patch_id)
        print(f"[RESULT] Patch {patch_id}: Detected={issue_detected}, Rollback={rollback}, Blocked={users_impacted_full}, Harmful={users_impacted_no_canary}")
        table8_rows.append([
            patch_id,
            issue_detected,
            rollback,
            users_impacted_full,
            users_impacted_no_canary
        ])
        total_blocked += users_impacted_full
        total_harmful += users_impacted_no_canary
    table8_rows.append([
        "Totals", "–", "–", total_blocked, total_harmful
    ])
    os.makedirs("table_results", exist_ok=True)
    with open("table_results/table8_canary_catch_rollback.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Patch ID", "Issue Detected at Canary?", "Rollback at Canary?", "Users Impacted (Full; from Replays)", "Users Impacted (No-Canary; from Replays)"])
        writer.writerows(table8_rows)
    print("Table 8 generated: table_results/table8_canary_catch_rollback.csv")

if __name__ == "__main__":
    main()
