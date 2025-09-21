from __future__ import annotations
import os
import random
from typing import List

import numpy as np
import torch

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import re
from pathlib import Path
"""
Utility functions for processing mathematical reasoning experiment data & plotting.

This module provides clean, modular functions for data loading, processing, 
and statistical analysis without excessive output.
"""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

def load_data(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and clean the CSV data.
    
    Args:
        filepath: Path to the CSV file
        verbose: Whether to print loading information
        
    Returns:
        Cleaned DataFrame
    """
    data = pd.read_csv(filepath)
    
    # Strip whitespace from string columns
    string_cols = data.select_dtypes(include=['object']).columns
    for col in string_cols:
        data[col] = data[col].astype(str).str.strip()
    
    # Ensure correct data types
    data['correct'] = data['correct'].astype(bool)
    data['snapshot'] = data['snapshot'].astype(int)
    data['seed'] = data['seed'].astype(int)
    
    if verbose:
        print(f"Loaded {len(data)} rows with {len(data.columns)} columns")
    
    return data


def get_summary_stats(data: pd.DataFrame) -> Dict:
    """Get comprehensive summary statistics."""
    return {
        'total_samples': len(data),
        'unique_models': data['model'].nunique(),
        'unique_sizes': data['size'].nunique(),
        'unique_snapshots': data['snapshot'].nunique(),
        'overall_accuracy': data['correct'].mean(),
        'operators': data['operator'].unique().tolist(),
        'number_formats': data['number_format'].unique().tolist(),
        'snapshot_range': (data['snapshot'].min(), data['snapshot'].max())
    }


def calculate_accuracy_by_group(data: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Calculate accuracy grouped by specified columns.
    
    Args:
        data: Input DataFrame
        group_cols: List of columns to group by
        
    Returns:
        DataFrame with accuracy metrics by group
    """
    grouped = data.groupby(group_cols).agg({
        'correct': ['count', 'sum', 'mean'],
        'id': 'count'
    }).round(4)
    
    grouped.columns = ['total_samples', 'correct_samples', 'accuracy', 'verification_count']
    return grouped.reset_index()


def get_accuracy_by_snapshot(data: pd.DataFrame) -> pd.DataFrame:
    """Get accuracy trends over training snapshots."""
    return calculate_accuracy_by_group(data, ['model', 'size', 'snapshot'])


def get_accuracy_by_operator(data: pd.DataFrame) -> pd.DataFrame:
    """Get accuracy by mathematical operator."""
    return calculate_accuracy_by_group(data, ['model', 'size', 'operator'])


def get_accuracy_by_difficulty(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze accuracy by problem difficulty (number magnitudes).
    
    Returns:
        Tuple of (difficulty_by_max, difficulty_by_sum) DataFrames
    """
    data_copy = data.copy()
    data_copy['max_num'] = data_copy[['num1', 'num2']].max(axis=1)
    data_copy['sum_nums'] = data_copy['num1'] + data_copy['num2']
    
    # Difficulty categories
    data_copy['difficulty_max'] = pd.cut(data_copy['max_num'], 
                                       bins=[0, 3, 6, 10, float('inf')], 
                                       labels=['1-3', '4-6', '7-10', '10+'])
    
    data_copy['difficulty_sum'] = pd.cut(data_copy['sum_nums'], 
                                       bins=[0, 6, 12, 20, float('inf')], 
                                       labels=['≤6', '7-12', '13-20', '>20'])
    
    # Group by difficulty
    diff_max = calculate_accuracy_by_group(data_copy, ['model', 'size', 'difficulty_max'])
    diff_sum = calculate_accuracy_by_group(data_copy, ['model', 'size', 'difficulty_sum'])
    
    return diff_max, diff_sum


def extract_prediction_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """Analyze common patterns in model predictions."""
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


def calculate_confidence_intervals(accuracies: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence intervals for accuracy measurements."""
    n = len(accuracies)
    mean_acc = np.mean(accuracies)
    std_err = np.std(accuracies, ddof=1) / np.sqrt(n)
    
    from scipy import stats
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin_error = t_critical * std_err
    return mean_acc - margin_error, mean_acc + margin_error


def statistical_significance_test(group1_acc: List[bool], group2_acc: List[bool]) -> Dict:
    """Perform statistical significance test between two accuracy groups."""
    from scipy.stats import chi2_contingency, fisher_exact
    
    # Create contingency table
    g1_correct = sum(group1_acc)
    g1_total = len(group1_acc)
    g2_correct = sum(group2_acc) 
    g2_total = len(group2_acc)
    
    contingency_table = np.array([
        [g1_correct, g1_total - g1_correct],
        [g2_correct, g2_total - g2_correct]
    ])
    
    # Chi-square test
    chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
    
    # Fisher's exact test
    odds_ratio, p_fisher = fisher_exact(contingency_table)
    
    return {
        'group1_accuracy': g1_correct / g1_total,
        'group2_accuracy': g2_correct / g2_total,
        'chi2_statistic': chi2,
        'chi2_p_value': p_chi2,
        'fisher_odds_ratio': odds_ratio,
        'fisher_p_value': p_fisher,
        'significant_at_05': min(p_chi2, p_fisher) < 0.05
    }


def format_model_name(model: str, size: str) -> str:
    """Format model names for consistent display in plots."""
    size_map = {'small': 'S', 'medium': 'M', 'large': 'L', 'xl': 'XL'}
    return f"{model.title()} {size_map.get(size, size.title())}"


def snapshot_to_tokens(snapshot: int) -> str:
    """Convert snapshot number to human-readable token count."""
    if snapshot >= 1000:
        return f"{snapshot//1000}K"
    else:
        return str(snapshot)