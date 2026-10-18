#!/usr/bin/env python3
"""
Visualize the confusion matrix from Table S1
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Read the confusion matrix
cm_file = Path(__file__).parent / 'table_s1_confusion_matrix.csv'
df = pd.read_csv(cm_file, index_col=0)

# Extract numeric values (remove 'Total' row/column for visualization)
cm_values = df.iloc[:-1, :-1].values.astype(int)
row_labels = df.index[:-1].tolist()
col_labels = df.columns[:-1].tolist()

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Heatmap with counts
ax1 = axes[0]
sns.heatmap(cm_values, annot=True, fmt='d', cmap='Blues', 
            xticklabels=col_labels, yticklabels=row_labels,
            cbar_kws={'label': 'Count'}, ax=ax1)
ax1.set_title('Confusion Matrix: PatchEval vs Independent Judge\n(Counts)', 
              fontsize=14, fontweight='bold')
ax1.set_xlabel('PatchEval Label', fontsize=12)
ax1.set_ylabel('Independent Judge Label', fontsize=12)

# Plot 2: Normalized by row (Independent Judge perspective)
cm_normalized = cm_values.astype(float) / cm_values.sum(axis=1, keepdims=True)
cm_normalized = cm_normalized * 100  # Convert to percentage

ax2 = axes[1]
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Oranges',
            xticklabels=col_labels, yticklabels=row_labels,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax2)
ax2.set_title('Confusion Matrix: Normalized by Row\n(Percentage of Independent Judge Predictions)', 
              fontsize=14, fontweight='bold')
ax2.set_xlabel('PatchEval Label', fontsize=12)
ax2.set_ylabel('Independent Judge Label', fontsize=12)

# Add text annotations for key metrics
total = cm_values.sum()
tn, fp = cm_values[0, 0], cm_values[0, 1]  # Independent: Failure
fn, tp = cm_values[1, 0], cm_values[1, 1]  # Independent: Success

# Calculate metrics
accuracy = (tn + tp) / total
precision_failure = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_failure = tn / (tn + fp) if (tn + fp) > 0 else 0
precision_success = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_success = tp / (tp + fn) if (tp + fn) > 0 else 0

false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

# Add summary text box
summary_text = f"""
Summary Metrics:
• Total: {total}
• Accuracy: {accuracy:.1%}
• Agreement: {accuracy:.1%}

For Independent: Failure (Attack Blocked):
• Precision: {precision_failure:.1%}
• Recall: {recall_failure:.1%}

For Independent: Success (Attack Succeeds):
• Precision: {precision_success:.1%}
• Recall: {recall_success:.1%}

Critical Issues:
• False Positives: {fp} ({false_positive_rate:.1%} of failures)
• False Negatives: {fn} ({false_negative_rate:.1%} of successes)
• True Positives: {tp} (only {tp/total:.1%} where both agree on success)
"""

fig.suptitle('Table S1: Safety Labeling Audit - Confusion Matrix Analysis', 
              fontsize=16, fontweight='bold', y=1.02)

# Add text box with summary
fig.text(0.5, -0.15, summary_text, ha='center', va='top', 
         fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_file = Path(__file__).parent / 'table_s1_confusion_matrix_visualization.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved visualization to: {output_file}")

# Also create a detailed breakdown table
breakdown = pd.DataFrame({
    'Metric': [
        'True Negatives (TN)',
        'False Positives (FP)',
        'False Negatives (FN)',
        'True Positives (TP)',
        'Total',
        'Accuracy',
        'False Positive Rate',
        'False Negative Rate',
        'Precision (Failure)',
        'Recall (Failure)',
        'Precision (Success)',
        'Recall (Success)'
    ],
    'Value': [
        tn,
        fp,
        fn,
        tp,
        total,
        f'{accuracy:.3f}',
        f'{false_positive_rate:.3f}',
        f'{false_negative_rate:.3f}',
        f'{precision_failure:.3f}',
        f'{recall_failure:.3f}',
        f'{precision_success:.3f}',
        f'{recall_success:.3f}'
    ],
    'Interpretation': [
        'Both agree: attack blocked',
        'Independent: blocked, PatchEval: succeeded (FALSE POSITIVE)',
        'Independent: succeeded, PatchEval: blocked (FALSE NEGATIVE)',
        'Both agree: attack succeeded',
        'Total samples',
        'Overall agreement rate',
        'Fraction of blocked attacks incorrectly marked as succeeded',
        'Fraction of successful attacks incorrectly marked as blocked',
        'When PatchEval says "failure", how often is Independent also "failure"',
        'When Independent says "failure", how often does PatchEval agree',
        'When PatchEval says "success", how often is Independent also "success"',
        'When Independent says "success", how often does PatchEval agree'
    ]
})

breakdown_file = Path(__file__).parent / 'table_s1_confusion_matrix_breakdown.csv'
breakdown.to_csv(breakdown_file, index=False)
print(f"Saved breakdown to: {breakdown_file}")

print("\n" + "="*80)
print("CONFUSION MATRIX ANALYSIS")
print("="*80)
print(f"\nKey Issue: {fp} false positives (Independent: Success → PatchEval: Failure)")
print(f"This means PatchEval is blocking {fp/(tn+fp)*100:.1f}% of attacks")
print(f"that Independent judge says should succeed (are successful attacks)")
print(f"\nOnly {tp} true positives where both agree attack succeeded ({tp/total:.1%})")
print(f"Agreement rate: {accuracy:.1%} (kappa would be low due to class imbalance)")

plt.show()
