"""
Clean, modular analysis runner for mathematical reasoning experiments.

This script provides a streamlined interface for generating all analyses
with minimal output and clear separation of concerns.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Import our modular components
import plot
import analysis_utils

class AnalysisRunner:
    """Main class for running comprehensive analysis."""
    
    def __init__(self, data_path: str, output_dir: Optional[str] = None, verbose: bool = False):
        """
        Initialize analysis runner.
        
        Args:
            data_path: Path to CSV data file
            output_dir: Output directory (defaults to current directory)
            verbose: Whether to print progress messages
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir) if output_dir else Path("../..")
        self.verbose = verbose
        self.data = None
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and clean the data."""
        if self.verbose:
            print("Loading data...")
        self.data = analysis_utils.load_data(self.data_path)
        if self.verbose:
            print(f"Loaded  {len(self.data)} rows with {len(self.data.columns)} columns")
        return self.data
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate all analysis metrics."""
        if self.data is None:
            self.load_data()
        
        if self.verbose:
            print("Calculating metrics...")
            
        self.results = {
            'summary_stats': utils.get_summary_stats(self.data),
            'accuracy_by_snapshot': utils.get_accuracy_by_snapshot(self.data),
            'accuracy_by_operator': utils.get_accuracy_by_operator(self.data),
            'difficulty_analysis': utils.get_accuracy_by_difficulty(self.data),
            'prediction_patterns': utils.extract_prediction_patterns(self.data)
        }
        
        return self.results
    
    def generate_plots(self) -> Dict[str, str]:
        """Generate all visualization plots."""
        if not self.results:
            self.calculate_metrics()
        
        if self.verbose:
            print("Generating plots...")
        
        plot_files = {}
        
        # Learning curves
        plot.plot_learning_curves(
            self.results['accuracy_by_snapshot'],
            save_path=self.output_dir / "learning_curves.png"
        )
        plot_files['learning_curves'] = "learning_curves.png"
        
        # Operator comparison
        plot.plot_operator_comparison(
            self.results['accuracy_by_operator'],
            save_path=self.output_dir / "operator_comparison.png"
        )
        plot_files['operator_comparison'] = "operator_comparison.png"
        
        # Difficulty analysis
        diff_max, diff_sum = self.results['difficulty_analysis']
        plot.plot_difficulty_analysis(
            diff_max, diff_sum,
            save_path=self.output_dir / "difficulty_analysis"
        )
        plot_files['difficulty_max'] = "difficulty_analysis_max_difficulty.png"
        plot_files['difficulty_sum'] = "difficulty_analysis_sum_difficulty.png"
        
        # Prediction patterns
        plot.plot_prediction_patterns(
            self.results['prediction_patterns'],
            save_path=self.output_dir / "prediction_patterns.png"
        )
        plot_files['prediction_patterns'] = "prediction_patterns.png"
        
        # Model scaling
        plot.plot_model_scaling(
            self.results['accuracy_by_snapshot'],
            save_path=self.output_dir / "model_scaling.png"
        )
        plot_files['model_scaling'] = "model_scaling.png"
        
        # Comprehensive dashboard
        plot.plot_comprehensive_dashboard(
            self.results['accuracy_by_snapshot'],
            self.results['accuracy_by_operator'],
            self.results['prediction_patterns'],
            save_path=self.output_dir / "comprehensive_dashboard.png"
        )
        plot_files['dashboard'] = "comprehensive_dashboard.png"
        
        # Publication figure
        plot.create_publication_figure(
            self.results['accuracy_by_snapshot'],
            self.results['accuracy_by_operator'],
            save_path=self.output_dir / "publication_figure.png"
        )
        plot_files['publication'] = "publication_figure.png"
        
        return plot_files
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        if not self.results:
            self.calculate_metrics()
        
        stats = self.results['summary_stats']
        
        # Get final performance metrics
        snapshot_data = self.results['accuracy_by_snapshot']
        latest_snapshot = snapshot_data.groupby(['model', 'size'])['snapshot'].max().reset_index()
        final_performance = snapshot_data.merge(latest_snapshot, on=['model', 'size', 'snapshot'])
        
        # Statistical comparisons
        medium_correct = self.data[self.data['size'] == 'medium']['correct'].tolist()
        large_correct = self.data[self.data['size'] == 'large']['correct'].tolist()
        
        add_correct = self.data[self.data['operator'] == '+']['correct'].tolist()
        sub_correct = self.data[self.data['operator'] == '-']['correct'].tolist()
        
        # Generate report content
        report = f"""# Mathematical Reasoning Experiment Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview
- Total samples: {stats['total_samples']:,}
- Models: {stats['unique_models']} ({', '.join([str(m) for m in self.data['model'].unique()])})
- Model sizes: {stats['unique_sizes']} ({', '.join(self.data['size'].unique())})
- Training snapshots: {stats['unique_snapshots']} (range: {stats['snapshot_range'][0]:,} to {stats['snapshot_range'][1]:,})
- Overall accuracy: {stats['overall_accuracy']:.3f} ({stats['overall_accuracy']*100:.1f}%)
- Operations: {', '.join(stats['operators'])}

## Performance Summary

### By Model Size
"""
        
        for _, row in final_performance.iterrows():
            report += f"- {row['model'].title()} {row['size'].title()}: {row['accuracy']*100:.1f}% (samples: {row['total_samples']:,})\n"
        
        ### By Operation
        report += "\n### By Operation\n"
        operator_data = self.results['accuracy_by_operator']
        for _, row in operator_data.iterrows():
            op_symbol = '+' if row['operator'] == '+' else '−'
            report += f"- {row['model'].title()} {row['size'].title()} ({op_symbol}): {row['accuracy']*100:.1f}%\n"
        
        ### Response Patterns
        report += "\n### Common Response Patterns\n"
        patterns = self.results['prediction_patterns']
        for _, row in patterns.iterrows():
            pattern_name = row['pattern'].replace('_', ' ').title()
            report += f"- {pattern_name}: {row['percentage']:.1f}% of responses (accuracy: {row['accuracy_when_present']:.1f}%)\n"
        
        ### Statistical Analysis
        report += "\n### Statistical Comparisons\n"
        
        if medium_correct and large_correct:
            sig_test = utils.statistical_significance_test(medium_correct, large_correct)
            report += f"- Model sizes (Medium vs Large):\n"
            report += f"  - Difference: {(sig_test['group2_accuracy'] - sig_test['group1_accuracy'])*100:.1f} percentage points\n"
            report += f"  - p-value: {sig_test['fisher_p_value']:.4f}\n"
            report += f"  - Significant: {'Yes' if sig_test['significant_at_05'] else 'No'}\n"
        
        if add_correct and sub_correct:
            sig_test_ops = utils.statistical_significance_test(add_correct, sub_correct)
            report += f"- Operations (Addition vs Subtraction):\n"
            report += f"  - Difference: {(sig_test_ops['group1_accuracy'] - sig_test_ops['group2_accuracy'])*100:.1f} percentage points\n"
            report += f"  - p-value: {sig_test_ops['fisher_p_value']:.4f}\n"
            report += f"  - Significant: {'Yes' if sig_test_ops['significant_at_05'] else 'No'}\n"
        
        ### Key Insights
        report += "\n### Key Insights\n"
        report += f"1. Very low overall accuracy ({stats['overall_accuracy']*100:.1f}%) indicates significant challenges\n"
        report += f"2. Model scaling shows minimal benefit\n"
        
        top_pattern = patterns.loc[patterns['percentage'].idxmax()]
        report += f"3. Primary failure mode: {top_pattern['pattern'].replace('_', ' ')} ({top_pattern['percentage']:.1f}% of responses)\n"
        report += f"4. Limited learning progression across training snapshots\n"
        
        ### Files Generated
        report += "\n## Generated Files\n"
        report += "- learning_curves.png: Training progress visualization\n"
        report += "- operator_comparison.png: Addition vs subtraction analysis\n"
        report += "- difficulty_analysis_max_difficulty.png: Performance by maximum operand\n"
        report += "- difficulty_analysis_sum_difficulty.png: Performance by operand sum\n"
        report += "- prediction_patterns.png: Response pattern analysis\n"
        report += "- model_scaling.png: Model size comparison\n"
        report += "- comprehensive_dashboard.png: Multi-panel overview\n"
        report += "- publication_figure.png: Publication-ready main results\n"
        report += "- analysis_report.txt: This summary report\n"
        
        # Save report
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        if self.verbose:
            print("🚀 Starting analysis pipeline")
        
        # Load data and calculate metrics
        self.load_data()
        self.calculate_metrics()
        
        # Generate all plots
        plot_files = self.generate_plots()
        
        # Generate report
        report = self.generate_report()
        
        # Save processed data
        if self.verbose:
            print("Saving processed data...")
        
        self.results['accuracy_by_snapshot'].to_csv(
            self.output_dir / "learning_curves_data.csv", index=False
        )
        self.results['accuracy_by_operator'].to_csv(
            self.output_dir / "operator_data.csv", index=False
        )
        self.results['prediction_patterns'].to_csv(
            self.output_dir / "patterns_data.csv", index=False
        )
        
        summary = {
            'files_generated': plot_files,
            'report_generated': True,
            'data_saved': True,
            'summary_stats': self.results['summary_stats']
        }
        
        if self.verbose:
            print("✅ Analysis complete")
            print(f"Overall accuracy: {self.results['summary_stats']['overall_accuracy']*100:.1f}%")
        return summary


def quick_analysis(data_path: str = "resultsImplicite.csv", 
                  output_dir: str = "./analysis_output/", 
                  verbose: bool = True) -> AnalysisRunner:
    """
    Quick analysis function for interactive use.
    
    Args:
        data_path: Path to CSV file
        output_dir: Output directory for results
        verbose: Whether to print progress
        
    Returns:
        AnalysisRunner instance with completed analysis
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run analysis
    runner = AnalysisRunner(data_path, output_dir, verbose=verbose)
    runner.run_full_analysis()
    
    return runner


def custom_analysis(data_path: str, analyses: list = None, output_dir: str = "./") -> Dict:
    """
    Run custom subset of analyses.
    
    Args:
        data_path: Path to CSV file
        analyses: List of analysis types to run
        output_dir: Output directory
        
    Returns:
        Dictionary with requested results
    """
    if analyses is None:
        analyses = ['summary', 'learning_curves', 'operators']
    
    runner = AnalysisRunner(data_path, output_dir, verbose=False)
    runner.load_data()
    runner.calculate_metrics()
    
    results = {}
    
    if 'summary' in analyses:
        results['summary'] = runner.results['summary_stats']
    
    if 'learning_curves' in analyses:
        plot.plot_learning_curves(
            runner.results['accuracy_by_snapshot'],
            save_path=Path(output_dir) / "learning_curves.png"
        )
        results['learning_curves'] = "learning_curves.png"
    
    if 'operators' in analyses:
        plot.plot_operator_comparison(
            runner.results['accuracy_by_operator'],
            save_path=Path(output_dir) / "operators.png"
        )
        results['operators'] = "operators.png"
    
    if 'patterns' in analyses:
        plot.plot_prediction_patterns(
            runner.results['prediction_patterns'],
            save_path=Path(output_dir) / "patterns.png"
        )
        results['patterns'] = "patterns.png"
    
    return results


def main():
    """Main entry point for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Mathematical Reasoning Analysis')
    parser.add_argument('--data', default='resultsImplicite.csv', help='Data file path')
    parser.add_argument('--output', default='./analysis_output/', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quick', action='store_true', help='Run quick analysis')
    
    args = parser.parse_args()
    
    if args.quick:
        runner = quick_analysis(args.data, args.output, args.verbose)
    else:
        runner = AnalysisRunner(args.data, args.output, args.verbose)
        runner.run_full_analysis()
    
    print(f"Results saved to: {args.output}")


from run_analysis import quick_analysis
if __name__ == "__main__":
    main()
    runner = quick_analysis("resultsImplicite.csv")
    
    
    