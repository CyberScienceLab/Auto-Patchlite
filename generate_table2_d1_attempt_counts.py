#!/usr/bin/env python3
"""
Generate Table 2 (D1: Attempt counts and dataflow from seed prompts to total evaluated attempts).

All counts are read from data files (audit or Table 7 CSVs). No hardcoded family names or N.

Usage:
  python generate_table2_d1_attempt_counts.py --source audit
  python generate_table2_d1_attempt_counts.py --source table7
  python generate_table2_d1_attempt_counts.py --source audit --variants 1 --rounds 1
  python generate_table2_d1_attempt_counts.py -o table_results/table2_d1.csv

Output: CSV with columns Attack family, Seed prompts, Variants/seed, Rounds, Attempts/round, Total attempts.
"""

import argparse
import csv
from pathlib import Path

DEFAULT_VARIANTS = 1
DEFAULT_ROUNDS = 1

# Default paths (relative to repo root); overridable via CLI
DEFAULT_AUDIT_CSV = "new_experiments/results/table_s1_per_family_metrics.csv"
DEFAULT_TABLE7_CSV = "table7/table7_coverage_across_families.csv"


def load_audit_families(path: Path):
    """Load (family_name, N) from table_s1_per_family_metrics.csv (columns Family, N)."""
    if not path.exists():
        raise FileNotFoundError(f"Audit CSV not found: {path}")
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            fam = (row.get("Family") or "").strip()
            n = (row.get("N") or "").strip()
            if fam and n and n.isdigit():
                rows.append((fam, int(n)))
    return rows


def load_table7_families(path: Path):
    """Load (attack_family, N_attempts) from table7 coverage CSV. N (#Attempts) = total evaluated."""
    if not path.exists():
        raise FileNotFoundError(f"Table 7 CSV not found: {path}")
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            name = (row.get("Attack Family") or "").strip()
            n_raw = (row.get("N (#Attempts)") or "").strip().replace(",", "")
            if name and n_raw and n_raw.isdigit():
                rows.append((name, int(n_raw)))
    return rows


def build_rows_from_audit(families: list, variants: int, rounds: int):
    """Families: [(name, N), ...]. Seed = N; total = N * variants * rounds."""
    rows = []
    for name, n_seed in families:
        n_total = n_seed * variants * rounds
        attempts_per_round = n_total // rounds if rounds else n_total
        rows.append({
            "Attack family": name,
            "Seed prompts": n_seed,
            "Variants/seed": variants,
            "Rounds": rounds,
            "Attempts/round": attempts_per_round,
            "Total attempts": n_total,
        })
    return rows


def build_rows_from_table7(families: list, variants: int, rounds: int):
    """Families: [(name, N_attempts), ...]. Table 7 only has total N; we use seed = N (1:1) when no separate seed data."""
    rows = []
    for name, n_total in families:
        n_seed = n_total  # 1:1 when no separate seed count
        attempts_per_round = n_total // rounds if rounds else n_total
        rows.append({
            "Attack family": name,
            "Seed prompts": n_seed,
            "Variants/seed": variants,
            "Rounds": rounds,
            "Attempts/round": attempts_per_round,
            "Total attempts": n_total,
        })
    return rows


def add_overall_row(rows: list, variants: int, rounds: int):
    total_seed = sum(r["Seed prompts"] for r in rows)
    total_attempt = sum(r["Total attempts"] for r in rows)
    rows.append({
        "Attack family": "Overall",
        "Seed prompts": total_seed,
        "Variants/seed": variants,
        "Rounds": rounds,
        "Attempts/round": "-",
        "Total attempts": total_attempt,
    })


def main():
    ap = argparse.ArgumentParser(description="Generate Table 2 (D1) attempt counts CSV")
    ap.add_argument("--source", choices=("audit", "table7"), default="audit",
                    help="Data source: audit (table_s1_per_family) or table7 (coverage)")
    ap.add_argument("--audit-csv", default=DEFAULT_AUDIT_CSV,
                    help="Path to audit per-family CSV (Family, N)")
    ap.add_argument("--table7-csv", default=DEFAULT_TABLE7_CSV,
                    help="Path to Table 7 coverage CSV (Attack Family, N (#Attempts))")
    ap.add_argument("--variants", type=int, default=DEFAULT_VARIANTS,
                    help="Variants per seed (default 1)")
    ap.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                    help="Rounds (default 1)")
    ap.add_argument("-o", "--output", default="table_results/table2_d1_attempt_counts.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    root = Path(__file__).parent.resolve()
    audit_path = root / args.audit_csv
    table7_path = root / args.table7_csv

    if args.source == "audit":
        families = load_audit_families(audit_path)
        rows = build_rows_from_audit(families, args.variants, args.rounds)
    else:
        families = load_table7_families(table7_path)
        rows = build_rows_from_table7(families, args.variants, args.rounds)

    add_overall_row(rows, args.variants, args.rounds)

    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Attack family", "Seed prompts", "Variants/seed", "Rounds", "Attempts/round", "Total attempts"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {out}")
    print("Preview:")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
