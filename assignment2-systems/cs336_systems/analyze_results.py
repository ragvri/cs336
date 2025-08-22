#!/usr/bin/env python3
"""
Analyze benchmark results from hyperparameter sweeps.
Aggregates JSON results into pandas DataFrames and generates LaTeX/Markdown tables.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import click


def load_benchmark_results(results_dir: str) -> List[Dict[Any, Any]]:
    """Load all benchmark JSON results from a directory."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory {results_dir} does not exist")
    
    json_files = list(results_path.glob("*.json"))
    # Filter out sweep_info.json
    json_files = [f for f in json_files if f.name != "sweep_info.json"]
    
    print(f"Found {len(json_files)} result files")
    
    results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Successfully loaded {len(results)} benchmark results")
    return results


def create_summary_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create a summary DataFrame from benchmark results."""
    
    rows = []
    
    for result in results:
        hyperparams = result['hyperparameters']
        system_info = result.get('system_info', {})
        
        # Base row with hyperparameters
        base_row = {
            'run_id': result['run_id'],
            'timestamp': result['timestamp'],
            'num_layers': hyperparams['num_layers'],
            'num_heads': hyperparams['num_heads'],
            'd_model': hyperparams['d_model'],
            'd_ff': hyperparams['d_ff'],
            'batch_size': hyperparams['batch_size'],
            'context_length': hyperparams['context_length'],
            'total_parameters': hyperparams['total_parameters'],
            'gpu_name': system_info.get('gpu_name'),
            'gpu_memory_gb': system_info.get('gpu_memory_gb'),
        }
        
        # Add results for each precision mode
        for precision_mode, precision_results in result['results'].items():
            row = base_row.copy()
            row['precision_mode'] = precision_mode
            row['forward_time'] = precision_results['forward_pass']['mean_time']
            row['forward_std'] = precision_results['forward_pass']['std_time']
            row['backward_time'] = precision_results['backward_pass']['mean_time']
            row['backward_std'] = precision_results['backward_pass']['std_time']
            row['optimizer_time'] = precision_results['optimizer_step']['mean_time']
            row['optimizer_std'] = precision_results['optimizer_step']['std_time']
            row['total_step_time'] = precision_results['total_step_time']['mean_time']
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add derived columns
    df['model_size_category'] = pd.cut(df['total_parameters'], 
                                     bins=[0, 50e6, 200e6, 500e6, float('inf')],
                                     labels=['Small (<50M)', 'Medium (50-200M)', 'Large (200-500M)', 'XLarge (>500M)'])
    
    df['throughput_samples_per_sec'] = df['batch_size'] / df['total_step_time']
    df['throughput_tokens_per_sec'] = df['throughput_samples_per_sec'] * df['context_length']
    
    return df


def create_speedup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with speedup comparisons between full and mixed precision."""
    
    # Filter to only configurations that have both precision modes
    config_cols = ['num_layers', 'num_heads', 'd_model', 'd_ff', 'batch_size', 'context_length']
    
    speedup_rows = []
    
    for config_vals, group in df.groupby(config_cols):
        if len(group['precision_mode'].unique()) == 2:  # Both full and mixed precision
            full_row = group[group['precision_mode'] == 'full'].iloc[0]
            mixed_row = group[group['precision_mode'] == 'mixed'].iloc[0]
            
            speedup_row = {
                'num_layers': config_vals[0],
                'num_heads': config_vals[1], 
                'd_model': config_vals[2],
                'd_ff': config_vals[3],
                'batch_size': config_vals[4],
                'context_length': config_vals[5],
                'total_parameters': full_row['total_parameters'],
                'model_size_category': full_row['model_size_category'],
                'forward_speedup': full_row['forward_time'] / mixed_row['forward_time'],
                'backward_speedup': full_row['backward_time'] / mixed_row['backward_time'],
                'total_speedup': full_row['total_step_time'] / mixed_row['total_step_time'],
                'throughput_speedup': mixed_row['throughput_samples_per_sec'] / full_row['throughput_samples_per_sec'],
                'full_forward_time': full_row['forward_time'],
                'mixed_forward_time': mixed_row['forward_time'],
                'full_backward_time': full_row['backward_time'],
                'mixed_backward_time': mixed_row['backward_time'],
            }
            
            speedup_rows.append(speedup_row)
    
    return pd.DataFrame(speedup_rows)


def generate_summary_tables(df: pd.DataFrame, output_dir: str):
    """Generate summary tables in LaTeX and Markdown formats."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. Performance by model size and precision
    perf_summary = df.groupby(['model_size_category', 'precision_mode']).agg({
        'forward_time': ['mean', 'std'],
        'backward_time': ['mean', 'std'], 
        'total_step_time': ['mean', 'std'],
        'throughput_samples_per_sec': ['mean', 'std'],
        'total_parameters': 'mean'
    }).round(4)
    
    # Flatten column names
    perf_summary.columns = ['_'.join(col).strip() for col in perf_summary.columns]
    perf_summary = perf_summary.reset_index()
    
    # Save performance summary tables
    with open(output_path / 'performance_summary.tex', 'w') as f:
        f.write(perf_summary.to_latex(index=False, caption="Performance Summary by Model Size and Precision Mode", 
                                    label="tab:performance_summary"))
    
    with open(output_path / 'performance_summary.md', 'w') as f:
        f.write("# Performance Summary by Model Size and Precision Mode\n\n")
        f.write(perf_summary.to_markdown(index=False))
    
    print(f"Performance summary tables saved to {output_path}")
    
    return perf_summary


def generate_speedup_tables(speedup_df: pd.DataFrame, output_dir: str):
    """Generate speedup analysis tables."""
    
    output_path = Path(output_dir)
    
    # Speedup by model size
    speedup_summary = speedup_df.groupby('model_size_category').agg({
        'forward_speedup': ['mean', 'std', 'min', 'max'],
        'backward_speedup': ['mean', 'std', 'min', 'max'],
        'total_speedup': ['mean', 'std', 'min', 'max'],
        'throughput_speedup': ['mean', 'std', 'min', 'max'],
        'total_parameters': 'mean'
    }).round(3)
    
    # Flatten column names
    speedup_summary.columns = ['_'.join(col).strip() for col in speedup_summary.columns]
    speedup_summary = speedup_summary.reset_index()
    
    # Save speedup tables
    with open(output_path / 'speedup_analysis.tex', 'w') as f:
        f.write(speedup_summary.to_latex(index=False, caption="Mixed Precision Speedup Analysis by Model Size", 
                                       label="tab:speedup_analysis"))
    
    with open(output_path / 'speedup_analysis.md', 'w') as f:
        f.write("# Mixed Precision Speedup Analysis by Model Size\n\n")
        f.write(speedup_summary.to_markdown(index=False))
        f.write("\n\n## Interpretation\n")
        f.write("- Values > 1.0 indicate mixed precision is faster\n")
        f.write("- Values < 1.0 indicate full precision is faster\n")
    
    # Detailed speedup table for specific configurations
    detailed_speedup = speedup_df[['num_layers', 'num_heads', 'd_model', 'batch_size', 'context_length',
                                  'forward_speedup', 'backward_speedup', 'total_speedup', 'throughput_speedup']].round(3)
    
    with open(output_path / 'detailed_speedup.tex', 'w') as f:
        f.write(detailed_speedup.to_latex(index=False, caption="Detailed Mixed Precision Speedup by Configuration", 
                                        label="tab:detailed_speedup"))
    
    with open(output_path / 'detailed_speedup.md', 'w') as f:
        f.write("# Detailed Mixed Precision Speedup by Configuration\n\n")
        f.write(detailed_speedup.to_markdown(index=False))
    
    print(f"Speedup analysis tables saved to {output_path}")
    
    return speedup_summary


def create_visualizations(df: pd.DataFrame, speedup_df: pd.DataFrame, output_dir: str):
    """Create visualization plots."""
    
    output_path = Path(output_dir)
    plt.style.use('seaborn-v0_8')
    
    # 1. Performance vs Model Size
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Analysis Across Model Sizes', fontsize=16)
    
    # Forward pass time
    sns.boxplot(data=df, x='model_size_category', y='forward_time', hue='precision_mode', ax=axes[0,0])
    axes[0,0].set_title('Forward Pass Time')
    axes[0,0].set_ylabel('Time (seconds)')
    
    # Backward pass time  
    sns.boxplot(data=df, x='model_size_category', y='backward_time', hue='precision_mode', ax=axes[0,1])
    axes[0,1].set_title('Backward Pass Time')
    axes[0,1].set_ylabel('Time (seconds)')
    
    # Total throughput
    sns.boxplot(data=df, x='model_size_category', y='throughput_samples_per_sec', hue='precision_mode', ax=axes[1,0])
    axes[1,0].set_title('Throughput (Samples/sec)')
    axes[1,0].set_ylabel('Samples per second')
    
    # Parameter count vs performance
    sns.scatterplot(data=df, x='total_parameters', y='total_step_time', hue='precision_mode', ax=axes[1,1])
    axes[1,1].set_title('Parameters vs Step Time')
    axes[1,1].set_xlabel('Total Parameters')
    axes[1,1].set_ylabel('Total Step Time (seconds)')
    axes[1,1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path / 'performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Speedup analysis
    if not speedup_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Mixed Precision Speedup Analysis', fontsize=16)
        
        # Forward speedup
        sns.boxplot(data=speedup_df, x='model_size_category', y='forward_speedup', ax=axes[0])
        axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        axes[0].set_title('Forward Pass Speedup')
        axes[0].set_ylabel('Speedup (Full/Mixed)')
        
        # Backward speedup
        sns.boxplot(data=speedup_df, x='model_size_category', y='backward_speedup', ax=axes[1]) 
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        axes[1].set_title('Backward Pass Speedup')
        axes[1].set_ylabel('Speedup (Full/Mixed)')
        
        # Overall speedup
        sns.boxplot(data=speedup_df, x='model_size_category', y='total_speedup', ax=axes[2])
        axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        axes[2].set_title('Total Step Speedup')
        axes[2].set_ylabel('Speedup (Full/Mixed)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'speedup_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_path}")


@click.command()
@click.option("--results_dir", default="results", help="Directory containing benchmark JSON files")
@click.option("--output_dir", default="analysis", help="Directory to save analysis outputs")
@click.option("--create_plots", is_flag=True, help="Create visualization plots")
def main(results_dir, output_dir, create_plots):
    """Analyze benchmark results and generate tables and plots."""
    
    # Load results
    print(f"Loading results from {results_dir}...")
    results = load_benchmark_results(results_dir)
    
    if not results:
        print("No results found. Exiting.")
        return
    
    # Create DataFrames
    print("Creating summary DataFrame...")
    df = create_summary_dataframe(results)
    print(f"Created DataFrame with {len(df)} rows")
    
    print("Creating speedup DataFrame...")
    speedup_df = create_speedup_dataframe(df)
    print(f"Created speedup DataFrame with {len(speedup_df)} rows")
    
    # Save raw DataFrames
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    df.to_csv(output_path / 'benchmark_results.csv', index=False)
    speedup_df.to_csv(output_path / 'speedup_results.csv', index=False)
    print(f"Raw DataFrames saved to {output_path}")
    
    # Generate tables
    print("Generating summary tables...")
    perf_summary = generate_summary_tables(df, output_dir)
    
    if not speedup_df.empty:
        print("Generating speedup tables...")
        speedup_summary = generate_speedup_tables(speedup_df, output_dir)
    else:
        print("No speedup comparisons available (need both precision modes)")
    
    # Create visualizations
    if create_plots:
        print("Creating visualizations...")
        create_visualizations(df, speedup_df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print(f"- Raw data: benchmark_results.csv, speedup_results.csv")
    print(f"- LaTeX tables: *.tex files")
    print(f"- Markdown tables: *.md files")
    if create_plots:
        print(f"- Plots: *.png files")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total benchmark runs: {len(df)}")
    print(f"Unique configurations: {len(df.groupby(['num_layers', 'num_heads', 'd_model', 'd_ff', 'batch_size', 'context_length']))}")
    print(f"Precision modes: {df['precision_mode'].unique()}")
    print(f"Model size range: {df['total_parameters'].min():,.0f} - {df['total_parameters'].max():,.0f} parameters")
    
    if not speedup_df.empty:
        print(f"\n=== SPEEDUP ANALYSIS ===")
        print(f"Average forward speedup: {speedup_df['forward_speedup'].mean():.2f}x")
        print(f"Average backward speedup: {speedup_df['backward_speedup'].mean():.2f}x")
        print(f"Average total speedup: {speedup_df['total_speedup'].mean():.2f}x")


if __name__ == "__main__":
    main()