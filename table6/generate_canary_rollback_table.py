import pandas as pd
import os

# Patch IDs from Table 6
PATCH_IDS = [
    "e797d27a",
    "e61ba3b3",
    "0cd819e5",
    "3801858c",
]

# User IDs for demonstration (U1, U2, U3, U4)
USER_IDS = [f"U{i+1}" for i in range(len(PATCH_IDS))]

rows = []
for i, patch_id in enumerate(PATCH_IDS):
    rows.append({
        "Patch ID": patch_id,
        "Issue Detected at Canary?": "Yes/No",
        "Rollback at Canary?": "Yes/No",
        "Users Impacted (Full; from Replays)": 0,
        "Users Impacted (No-Canary; from Replays)": USER_IDS[i],
    })

# Add totals row
rows.append({
    "Patch ID": "Totals",
    "Issue Detected at Canary?": "-",
    "Rollback at Canary?": "-",
    "Users Impacted (Full; from Replays)": 0,
    "Users Impacted (No-Canary; from Replays)": ">=ΣUi",
})

df = pd.DataFrame(rows)
os.makedirs("table6", exist_ok=True)
df.to_csv("table6/canary_rollback_table.csv", index=False)
print("[Table6] Canary/Rollback table written to table6/canary_rollback_table.csv")