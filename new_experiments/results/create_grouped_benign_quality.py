#!/usr/bin/env python3
"""
Create a grouped version of benign_quality.csv by Category (JBB, InTheWild) and Family
"""

import csv
import sys
from collections import defaultdict

def categorize_source(source):
    """Categorize source into Category and Family"""
    if source.startswith('JailbreakBench'):
        return 'JBB', 'benign'
    elif source.startswith('InTheWild'):
        # Extract the config name (e.g., "regular_2023_05_07" from "InTheWild_regular_2023_05_07")
        parts = source.split('_', 1)
        if len(parts) > 1:
            family = parts[1]  # e.g., "regular_2023_05_07"
        else:
            family = 'unknown'
        return 'InTheWild', family
    else:
        return 'Other', source

def parse_float(value):
    """Parse float value, handling empty strings"""
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None

def main():
    input_file = 'benign_quality.csv'
    output_file = 'benign_quality_by_category_family.csv'
    
    # Read and group data
    grouped_data = defaultdict(lambda: defaultdict(lambda: {'values': [], 'count': 0}))
    metric_columns = []
    
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        metric_columns = [col for col in reader.fieldnames if col != 'source']
        
        for row in reader:
            source = row['source']
            category, family = categorize_source(source)
            
            for metric in metric_columns:
                key = f"{category}_{family}"
                value = parse_float(row[metric])
                if value is not None:
                    grouped_data[metric][key]['values'].append(value)
                    grouped_data[metric][key]['count'] += 1
    
    # Calculate means and prepare output
    output_rows = []
    categories_families = set()
    
    for metric in metric_columns:
        for key in grouped_data[metric]:
            categories_families.add(key)
    
    # Sort categories_families for consistent output
    categories_families = sorted(categories_families)
    
    for key in categories_families:
        category, family = key.split('_', 1)
        row = {'Category': category, 'Family': family}
        
        for metric in metric_columns:
            metric_key = f"{category}_{family}"
            if metric_key in grouped_data[metric]:
                values = grouped_data[metric][metric_key]['values']
                count = grouped_data[metric][metric_key]['count']
                if values:
                    mean_val = sum(values) / len(values)
                    row[metric] = f"{mean_val:.6f}"
                    row[f'{metric}_n'] = count
                else:
                    row[metric] = ''
                    row[f'{metric}_n'] = 0
            else:
                row[metric] = ''
                row[f'{metric}_n'] = 0
        
        output_rows.append(row)
    
    # Write output
    if output_rows:
        fieldnames = ['Category', 'Family']
        for metric in metric_columns:
            fieldnames.append(metric)
            fieldnames.append(f'{metric}_n')
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        
        print(f"Created {output_file}")
        print(f"\nSummary:")
        print(f"Total rows: {len(output_rows)}")
        print(f"\nFirst few rows:")
        for i, row in enumerate(output_rows[:5]):
            print(f"  {row['Category']} - {row['Family']}: {len([v for v in row.values() if v != ''])} metrics")
    else:
        print("No data to process")

if __name__ == '__main__':
    main()
