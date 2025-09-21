"""
Research-grade plotting functions for mathematical reasoning experiment analysis.

Clean, modular plotting functions with minimal output and professional styling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set research-grade style defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png'
})

# Color palette for consistency
COLORS = {
    'pythia_medium': '#1f77b4',
    'pythia_large': '#ff7f0e',
    'addition': '#2ca02c',
    'subtraction': '#d62728',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'neutral': '#7f7f7f'
}


def plot_learning_curves(data: pd.DataFrame, figsize: Tuple[int, int] = (12, 8), 
                        save_path: Optional[str] = None) -> plt.Figure:
    """Plot learning curves showing accuracy vs training snapshots."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for (model, size), group in data.groupby(['model', 'size']):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        
        group_sorted = group.sort_values('snapshot')
        
        ax.plot(group_sorted['snapshot'], group_sorted['accuracy'] * 100, 
               marker='o', linewidth=2.5, markersize=6, 
               color=color, label=f'{model.title()} {size.title()}',
               markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
    
    ax.set_xlabel('Training Snapshot')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Learning Curves: Accuracy vs Training Progress')
    
    ax.ticklabel_format(style='plain', axis='x')
    ax.set_xlim(left=0)
    ax.set_ylim(0, max(data['accuracy'].max() * 100 + 5, 100))
    
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_operator_comparison(data: pd.DataFrame, figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
    """Plot accuracy comparison by mathematical operator."""
    fig, ax = plt.subplots(figsize=figsize)
    
    pivot_data = data.pivot_table(index='operator', columns=['model', 'size'], 
                                 values='accuracy', aggfunc='mean')
    
    x_pos = np.arange(len(pivot_data.index))
    width = 0.35
    
    models_sizes = list(pivot_data.columns)
    for i, (model, size) in enumerate(models_sizes):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        
        values = pivot_data[(model, size)] * 100
        bars = ax.bar(x_pos + i * width - width/2, values, width, 
                     label=f'{model.title()} {size.title()}',
                     color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Mathematical Operator')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Mathematical Operator')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['+' if op == '+' else '−' for op in pivot_data.index])
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, max(pivot_data.values.flatten()) * 110)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_difficulty_analysis(diff_max: pd.DataFrame, diff_sum: pd.DataFrame, 
                           figsize: Tuple[int, int] = (10, 6),
                           save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Figure]:
    """Plot accuracy by problem difficulty (two different metrics)."""
    
    # Plot 1: Difficulty by maximum number
    fig1, ax1 = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(diff_max['difficulty_max'].unique()))
    width = 0.35
    
    for i, (model, size) in enumerate(diff_max[['model', 'size']].drop_duplicates().values):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        
        subset = diff_max[(diff_max['model'] == model) & (diff_max['size'] == size)]
        values = subset.sort_values('difficulty_max')['accuracy'] * 100
        
        bars = ax1.bar(x_pos + i * width - width/2, values, width,
                      label=f'{model.title()} {size.title()}',
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Difficulty (Maximum Number in Problem)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy by Problem Difficulty: Maximum Number')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sorted(diff_max['difficulty_max'].unique()))
    ax1.legend()
    ax1.set_ylim(0, max(diff_max['accuracy']) * 110)
    
    # Plot 2: Difficulty by sum of numbers
    fig2, ax2 = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(diff_sum['difficulty_sum'].unique()))
    
    for i, (model, size) in enumerate(diff_sum[['model', 'size']].drop_duplicates().values):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        
        subset = diff_sum[(diff_sum['model'] == model) & (diff_sum['size'] == size)]
        values = subset.sort_values('difficulty_sum')['accuracy'] * 100
        
        bars = ax2.bar(x_pos + i * width - width/2, values, width,
                      label=f'{model.title()} {size.title()}',
                      color=color, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Difficulty (Sum of Numbers in Problem)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Problem Difficulty: Sum of Numbers')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sorted(diff_sum['difficulty_sum'].unique()))
    ax2.legend()
    ax2.set_ylim(0, max(diff_sum['accuracy']) * 110)
    
    plt.tight_layout()
    
    if save_path:
        fig1.savefig(f"{save_path}_max_difficulty.png", dpi=300, bbox_inches='tight')
        fig2.savefig(f"{save_path}_sum_difficulty.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)
    
    return fig1, fig2


def plot_prediction_patterns(patterns_df: pd.DataFrame, figsize: Tuple[int, int] = (14, 6),
                           save_path: Optional[str] = None) -> plt.Figure:
    """Plot analysis of prediction patterns and failure modes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Frequency of patterns
    bars1 = ax1.barh(patterns_df['pattern'], patterns_df['percentage'],
                    color=COLORS['primary'], alpha=0.7, edgecolor='black')
    
    for bar, pct in zip(bars1, patterns_df['percentage']):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10)
    
    ax1.set_xlabel('Percentage of Responses')
    ax1.set_title('Frequency of Response Patterns')
    ax1.set_xlim(0, patterns_df['percentage'].max() * 1.1)
    
    # Plot 2: Accuracy when pattern is present
    bars2 = ax2.barh(patterns_df['pattern'], patterns_df['accuracy_when_present'],
                    color=COLORS['secondary'], alpha=0.7, edgecolor='black')
    
    for bar, acc in zip(bars2, patterns_df['accuracy_when_present']):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=10)
    
    ax2.set_xlabel('Accuracy When Pattern Present (%)')
    ax2.set_title('Accuracy by Response Pattern')
    ax2.set_xlim(0, max(patterns_df['accuracy_when_present'].max() * 1.1, 100))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_model_scaling(data: pd.DataFrame, figsize: Tuple[int, int] = (8, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
    """Plot model scaling analysis (accuracy vs model size)."""
    fig, ax = plt.subplots(figsize=figsize)
    
    sizes_order = ['medium', 'large']
    size_acc = data.groupby('size')['accuracy'].mean() * 100
    size_acc = size_acc.reindex(sizes_order)
    
    bars = ax.bar(range(len(size_acc)), size_acc.values, 
                 color=[COLORS[f'pythia_{size}'] for size in sizes_order],
                 alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, size_acc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Scaling: Accuracy vs Model Size')
    ax.set_xticks(range(len(sizes_order)))
    ax.set_xticklabels([size.title() for size in sizes_order])
    ax.set_ylim(0, max(size_acc.values) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_comprehensive_dashboard(learning_data: pd.DataFrame, operator_data: pd.DataFrame, 
                               patterns_data: pd.DataFrame, figsize: Tuple[int, int] = (16, 12),
                               save_path: Optional[str] = None) -> plt.Figure:
    """Create a comprehensive dashboard with multiple subplots."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Learning curves
    ax1 = fig.add_subplot(gs[0, 0])
    for (model, size), group in learning_data.groupby(['model', 'size']):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        group_sorted = group.sort_values('snapshot')
        ax1.plot(group_sorted['snapshot'], group_sorted['accuracy'] * 100,
                marker='o', linewidth=2, markersize=4, color=color,
                label=f'{model.title()} {size.title()}')
    ax1.set_xlabel('Training Snapshot')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('A) Learning Curves')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy by operator
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_data = operator_data.pivot_table(index='operator', columns=['model', 'size'],
                                         values='accuracy', aggfunc='mean')
    x_pos = np.arange(len(pivot_data.index))
    width = 0.35
    
    for i, (model, size) in enumerate(pivot_data.columns):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        values = pivot_data[(model, size)] * 100
        ax2.bar(x_pos + i * width - width/2, values, width,
               color=color, alpha=0.8, label=f'{model.title()} {size.title()}')
    
    ax2.set_xlabel('Operator')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('B) Accuracy by Operator')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['+' if op == '+' else '−' for op in pivot_data.index])
    ax2.legend(fontsize=9)
    
    # 3. Model scaling
    ax3 = fig.add_subplot(gs[1, 0])
    sizes_order = ['medium', 'large']
    size_acc = learning_data.groupby('size')['accuracy'].mean() * 100
    size_acc = size_acc.reindex(sizes_order)
    
    bars = ax3.bar(range(len(size_acc)), size_acc.values,
                  color=[COLORS[f'pythia_{size}'] for size in sizes_order],
                  alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, size_acc.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax3.set_xlabel('Model Size')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('C) Model Scaling')
    ax3.set_xticks(range(len(sizes_order)))
    ax3.set_xticklabels([size.title() for size in sizes_order])
    
    # 4. Prediction patterns
    ax4 = fig.add_subplot(gs[1, 1])
    top_patterns = patterns_data.nlargest(5, 'percentage')
    bars = ax4.barh(range(len(top_patterns)), top_patterns['percentage'],
                   color=COLORS['accent'], alpha=0.7)
    
    ax4.set_xlabel('Percentage (%)')
    ax4.set_ylabel('Response Pattern')
    ax4.set_title('D) Common Response Patterns')
    ax4.set_yticks(range(len(top_patterns)))
    ax4.set_yticklabels(top_patterns['pattern'], fontsize=9)
    
    for bar, pct in zip(bars, top_patterns['percentage']):
        ax4.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.suptitle('Mathematical Reasoning Experiment Results', fontsize=20, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig


def create_publication_figure(learning_data: pd.DataFrame, operator_data: pd.DataFrame,
                            figsize: Tuple[int, int] = (12, 5), 
                            save_path: Optional[str] = None) -> plt.Figure:
    """Create a publication-ready figure with learning curves and operator comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Learning curves
    for (model, size), group in learning_data.groupby(['model', 'size']):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        group_sorted = group.sort_values('snapshot')
        
        ax1.plot(group_sorted['snapshot'], group_sorted['accuracy'] * 100,
                marker='o', linewidth=2.5, markersize=6, color=color,
                label=f'{model.title()} {size.title()}',
                markerfacecolor='white', markeredgewidth=2, markeredgecolor=color)
    
    ax1.set_xlabel('Training Snapshot')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Learning Curves')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Operator comparison
    pivot_data = operator_data.pivot_table(index='operator', columns=['model', 'size'],
                                         values='accuracy', aggfunc='mean')
    x_pos = np.arange(len(pivot_data.index))
    width = 0.35
    
    for i, (model, size) in enumerate(pivot_data.columns):
        model_key = f"{model}_{size}"
        color = COLORS.get(model_key, COLORS['primary'])
        values = pivot_data[(model, size)] * 100
        
        bars = ax2.bar(x_pos + i * width - width/2, values, width,
                      color=color, alpha=0.8, edgecolor='black', linewidth=1,
                      label=f'{model.title()} {size.title()}')
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Mathematical Operator')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy by Operator')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['+' if op == '+' else '−' for op in pivot_data.index])
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return fig