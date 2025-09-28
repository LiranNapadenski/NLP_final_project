#!/usr/bin/env python3
"""
Simple standalone analysis script for mathematical reasoning experiments.
No external module dependencies - everything in one file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import re
from pathlib import Path

# Set up plotting style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

COLORS = {
    'pythia_medium': '#1f77b4',
    'pythia_large': '#ff7f0e'
}

def load_and_clean_data(filepath='resultsImplicite.csv'):
    """Load and clean the data."""
    print(f"Loading data from {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return None
    
    data = pd.read_csv(filepath)
    
    # Clean string columns
    string_cols = data.select_dtypes(include=['object']).columns
    for col in string_cols:
        data[col] = data[col].astype(str).str.strip()
    
    # Ensure correct data types
    data['correct'] = data['correct'].astype(bool)
    data['snapshot'] = data['snapshot'].astype(int)
    
    print(f"✓ Loaded {len(data)} rows")
    return data

def calculate_accuracy_by_snapshot(data):
    """Calculate accuracy by training snapshot."""
    accuracy_data = data.groupby(['model', 'size', 'snapshot']).agg({
        'correct': ['count', 'sum', 'mean']
    }).round(4)
    
    accuracy_data.columns = ['total_samples', 'correct_samples', 'accuracy']
    return accuracy_data.reset_index()

def calculate_accuracy_by_operator(data):
    """Calculate accuracy by mathematical operator."""
    operator_data = data.groupby(['model', 'size', 'operator']).agg({
        'correct': ['count', 'sum', 'mean']
    }).round(4)
    
    operator_data.columns = ['total_samples', 'correct_samples', 'accuracy']
    return operator_data.reset_index()

def analyze_prediction_patterns(data):
    """Analyze common patterns in predictions."""
    patterns = {
        'i_dont_know': r"(?i)(i don't know|i'm not sure|unsure|don't know)",
        'repetition': r'(.+?)\1{2,}',
        'no_answer': r'^$|^\s*$',
        'off_topic': r'(?i)(sorry|cannot|unable|can\'t)',
        'numeric_answer': r'\b\d+\b',
    }
    
    pattern_results = []
    for pattern_name, regex in patterns.items():
        matches = data['answer_only'].str.contains(regex, regex=True, na=False)
        pattern_results.append({
            'pattern': pattern_name,
            'count': matches.sum(),
            'percentage': matches.mean() * 100,
            'accuracy_when_present': data[matches]['correct'].mean() * 100 if matches.any() else 0
        })
    
    return pd.DataFrame(pattern_results)

def create_output_directory(base_dir="analysis_output"):
    """Create output directory."""
    output_dir = Path(base_dir)
    output_dir.mkdir(exist_ok=True)
    return output_dir

def plot_learning_curves(accuracy_data, output_dir):
    """Plot learning curves."""
    plt.figure(figsize=(12, 8))
    
    for (model, size), group in accuracy_data.groupby(['model', 'size']):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, '#1f77b4')
        group_sorted = group.sort_values('snapshot')
        
        plt.plot(group_sorted['snapshot'], group_sorted['accuracy'] * 100,
               marker='o', linewidth=2.5, markersize=6, color=color,
               label=f'{model.title()} {size.title()}',
               markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
    
    plt.xlabel('Training Snapshot')
    plt.ylabel('Accuracy (%)')
    plt.title('Learning Curves: Accuracy vs Training Progress')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    plt.tight_layout()
    save_path = output_dir / "learning_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_operator_comparison(operator_data, output_dir):
    """Plot accuracy by operator."""
    plt.figure(figsize=(10, 6))
    
    pivot_data = operator_data.pivot_table(index='operator', columns=['model', 'size'],
                                         values='accuracy', aggfunc='mean')
    
    x_pos = np.arange(len(pivot_data.index))
    width = 0.35
    
    for i, (model, size) in enumerate(pivot_data.columns):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, '#1f77b4')
        values = pivot_data[(model, size)] * 100
        
        bars = plt.bar(x_pos + i * width - width/2, values, width,
                     color=color, alpha=0.8, edgecolor='black', linewidth=1,
                     label=f'{model.title()} {size.title()}')
        
        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.xlabel('Mathematical Operator')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Mathematical Operator')
    plt.xticks(x_pos, ['+' if op == '+' else '−' for op in pivot_data.index])
    plt.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    save_path = output_dir / "operator_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_prediction_patterns(patterns_df, output_dir):
    """Plot prediction patterns analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Frequency
    bars1 = ax1.barh(patterns_df['pattern'], patterns_df['percentage'],
                    color='#1f77b4', alpha=0.7, edgecolor='black')
    
    for bar, pct in zip(bars1, patterns_df['percentage']):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10)
    
    ax1.set_xlabel('Percentage of Responses')
    ax1.set_title('Frequency of Response Patterns')
    ax1.set_xlim(0, patterns_df['percentage'].max() * 1.1)
    
    # Plot 2: Accuracy when present
    bars2 = ax2.barh(patterns_df['pattern'], patterns_df['accuracy_when_present'],
                    color='#ff7f0e', alpha=0.7, edgecolor='black')
    
    for bar, acc in zip(bars2, patterns_df['accuracy_when_present']):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=10)
    
    ax2.set_xlabel('Accuracy When Pattern Present (%)')
    ax2.set_title('Accuracy by Response Pattern')
    ax2.set_xlim(0, max(patterns_df['accuracy_when_present'].max() * 1.1, 100))
    
    plt.tight_layout()
    save_path = output_dir / "prediction_patterns.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_model_scaling(accuracy_data, output_dir):
    """Plot model scaling analysis."""
    plt.figure(figsize=(8, 6))
    
    sizes_order = ['medium', 'large']
    size_acc = accuracy_data.groupby('size')['accuracy'].mean() * 100
    size_acc = size_acc.reindex(sizes_order)
    
    bars = plt.bar(range(len(size_acc)), size_acc.values,
                 color=[COLORS[f'pythia_{size}'] for size in sizes_order],
                 alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, size_acc.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xlabel('Model Size')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Scaling: Accuracy vs Model Size')
    plt.xticks(range(len(sizes_order)), [size.title() for size in sizes_order])
    plt.ylim(0, max(size_acc.values) * 1.2)
    
    plt.tight_layout()
    save_path = output_dir / "model_scaling.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def generate_analysis_report(data, accuracy_data, operator_data, patterns_df, output_dir):
    """Generate comprehensive analysis report."""
    
    # Basic statistics
    total_samples = len(data)
    overall_accuracy = data['correct'].mean()
    
    # Final performance by model
    latest_snapshot = accuracy_data.groupby(['model', 'size'])['snapshot'].max().reset_index()
    final_performance = accuracy_data.merge(latest_snapshot, on=['model', 'size', 'snapshot'])
    
    report = f"""# Mathematical Reasoning Experiment Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview
- Total samples: {total_samples:,}
- Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)
- Models: {', '.join(data['model'].unique())}
- Model sizes: {', '.join(data['size'].unique())}
- Operations: {', '.join(data['operator'].unique())}
- Snapshot range: {data['snapshot'].min():,} to {data['snapshot'].max():,}

## Performance Summary

### By Model Size
"""
    
    for _, row in final_performance.iterrows():
        report += f"- {row['model'].title()} {row['size'].title()}: {row['accuracy']*100:.1f}% (samples: {row['total_samples']:,})\n"
    
    report += "\n### By Operation\n"
    for _, row in operator_data.iterrows():
        op_symbol = '+' if row['operator'] == '+' else '−'
        report += f"- {row['model'].title()} {row['size'].title()} ({op_symbol}): {row['accuracy']*100:.1f}%\n"
    
    report += "\n### Common Response Patterns\n"
    for _, row in patterns_df.iterrows():
        pattern_name = row['pattern'].replace('_', ' ').title()
        report += f"- {pattern_name}: {row['percentage']:.1f}% of responses (accuracy: {row['accuracy_when_present']:.1f}%)\n"
    
    report += "\n### Key Insights\n"
    report += f"1. Very low overall accuracy ({overall_accuracy*100:.1f}%) indicates significant challenges\n"
    report += f"2. Model scaling shows minimal improvement\n"
    
    top_pattern = patterns_df.loc[patterns_df['percentage'].idxmax()]
    report += f"3. Primary failure mode: {top_pattern['pattern'].replace('_', ' ')} ({top_pattern['percentage']:.1f}% of responses)\n"
    report += f"4. Limited learning progression across training snapshots\n"
    
    report += "\n## Generated Files\n"
    report += "- learning_curves.png: Training progress visualization\n"
    report += "- operator_comparison.png: Addition vs subtraction analysis\n"
    report += "- prediction_patterns.png: Response pattern analysis\n"
    report += "- model_scaling.png: Model size comparison\n"
    report += "- analysis_report.txt: This summary report\n"
    
    # Save report
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_path}")
    return report

def main():
    """Run the complete analysis."""
    print("🚀 Starting Mathematical Reasoning Analysis")
    print("=" * 50)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Load data
    data = load_and_clean_data()
    if data is None:
        return
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    accuracy_data = calculate_accuracy_by_snapshot(data)
    operator_data = calculate_accuracy_by_operator(data)
    patterns_df = analyze_prediction_patterns(data)
    
    # Generate plots
    print("\n📈 Generating visualizations...")
    plot_learning_curves(accuracy_data, output_dir)
    plot_operator_comparison(operator_data, output_dir)
    plot_prediction_patterns(patterns_df, output_dir)
    plot_model_scaling(accuracy_data, output_dir)
    
    # Generate report
    print("\n📋 Generating analysis report...")
    generate_analysis_report(data, accuracy_data, operator_data, patterns_df, output_dir)
    
    # Save processed data
    print("\n💾 Saving processed data...")
    accuracy_data.to_csv(output_dir / "learning_curves_data.csv", index=False)
    operator_data.to_csv(output_dir / "operator_data.csv", index=False)
    patterns_df.to_csv(output_dir / "patterns_data.csv", index=False)
    
    print("\n" + "=" * 50)
    print("✅ Analysis Complete!")
    print(f"📁 All files saved to: {output_dir.absolute()}")
    print(f"📊 Overall accuracy: {data['correct'].mean()*100:.1f}%")
    
    # Show top failure pattern
    top_pattern = patterns_df.loc[patterns_df['percentage'].idxmax()]
    print(f"🔍 Most common response pattern: {top_pattern['pattern']} ({top_pattern['percentage']:.1f}%)")

if __name__ == "__main__":
    main()