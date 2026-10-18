#!/bin/bash
# Run all table generation scripts and save outputs to table_results/

set -e

python3 generate_table6_attack_patch_sequence.py
python3 generate_table7_coverage_across_families.py
python3 generate_table8_canary_catch_rollback.py

echo "All tables generated in table_results/"
