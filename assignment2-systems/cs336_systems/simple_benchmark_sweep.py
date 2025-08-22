#!/usr/bin/env python3
"""
Simple benchmark sweep runner - no Slurm required.
Runs benchmarks sequentially for all model configurations.
"""

import sys
from pathlib import Path
from datetime import datetime
import click

# Add the cs336_systems directory to the path so we can import modules
sys.path.append(str(Path(__file__).parent))

from cs336_systems.model_configs import get_default_model_configs


@click.command()
@click.option("--results_dir", default="results", help="Directory to save benchmark results")
@click.option("--warmup_steps", default=20, help="Warmup steps for benchmarking")
@click.option("--benchmark_steps", default=50, help="Benchmark steps for benchmarking")
@click.option("--batch_size", default=4, help="Batch size for benchmarking")
@click.option("--context_lengths", default="128,256,512,1024", help="Comma-separated context lengths to test")
@click.option("--models", default="all", help="Comma-separated model names (e.g., 'small,medium') or 'all'")
@click.option("--memory_profile", is_flag=True, help="Enable detailed CUDA memory profiling (writes pickle snapshots).")
@click.option(
    "--memory_profile_dir",
    default=None,
    help="Directory to store memory snapshot pickle files (default: results_dir or CWD).",
)
@click.option(
    "--run_mode",
    type=click.Choice(["forward", "train"]),
    default="train",
    help="Run only forward pass (inference) or full training step (forward+backward+optimizer).",
)
@click.option(
    "--memory_profile_step",
    default=1,
    show_default=True,
    help="Which (1-indexed) benchmark step to capture a memory snapshot for when memory profiling is enabled.",
)
@click.option(
    "--precision",
    type=click.Choice(["both", "full", "mixed"]),
    default="both",
    help="Which precision modes to benchmark (both, full, mixed).",
)
def main(
    results_dir: str,
    warmup_steps: int,
    benchmark_steps: int,
    batch_size: int,
    context_lengths: str,
    models: str,
    memory_profile: bool,
    memory_profile_dir: str | None,
    run_mode: str,
    memory_profile_step: int,
    precision: str,
) -> None:
    """Run benchmark sweep for all model configurations sequentially."""

    # Create results directory
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    # Get model configurations
    all_configs = get_default_model_configs()

    # Filter models if specified
    if models != "all":
        model_names = [name.strip() for name in models.split(",")]
        configs = [config for config in all_configs if config.name in model_names]
        if not configs:
            print(f"No matching models found. Available: {[c.name for c in all_configs]}")
            return
    else:
        configs = all_configs

    # Parse context lengths
    context_length_list = [int(cl.strip()) for cl in context_lengths.split(",")]

    total_jobs = len(configs) * len(context_length_list)
    print(
        f"Running benchmarks for {len(configs)} model configurations × {len(context_length_list)} context lengths = {total_jobs} total jobs"
    )
    print(f"Models: {[c.name for c in configs]}")
    print(f"Context lengths: {context_length_list}")
    print(f"Results will be saved to: {results_path}")
    print(
        f"Settings: batch_size={batch_size}, warmup_steps={warmup_steps}, benchmark_steps={benchmark_steps}, memory_profile={memory_profile}, run_mode={run_mode}, memory_profile_step={memory_profile_step}, precision={precision}"
    )
    if memory_profile_dir:
        print(f"Memory profile dir (requested): {memory_profile_dir}")
    print("=" * 80)

    successful = 0
    failed = 0
    job_num = 0

    for config in configs:
        for context_length in context_length_list:
            job_num += 1
            print(f"\n[{job_num}/{total_jobs}] Benchmarking {config.name} with context_length={context_length}...")
            print(f"  Config: {config}")
            print(f"  Context length: {context_length}")

            # Create output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = results_path / f"{config.name}_ctx{context_length}_benchmark_{timestamp}.json"

            try:
                # Run the benchmark as subprocess
                import subprocess

                cmd = [
                    "uv",
                    "run",
                    "cs336_systems/benchmarking_script.py",
                    "--num_layers",
                    str(config.num_layers),
                    "--num_heads",
                    str(config.num_heads),
                    "--d_model",
                    str(config.d_model),
                    "--d_ff",
                    str(config.d_ff),
                    "--batch_size",
                    str(batch_size),
                    "--context_length",
                    str(context_length),
                    "--warmup_steps",
                    str(warmup_steps),
                    "--benchmark_steps",
                    str(benchmark_steps),
                    "--output_file",
                    str(output_file),
                    "--run_mode",
                    run_mode,
                    "--precision",
                    precision,
                ]

                if memory_profile:
                    cmd.append("--memory_profile")
                    if memory_profile_dir:
                        cmd.extend(["--memory_profile_dir", memory_profile_dir])
                    else:
                        # Use results directory by default for sweep if not provided
                        cmd.extend(["--memory_profile_dir", str(results_path)])
                    cmd.extend(["--memory_profile_step", str(memory_profile_step)])

                # Don't add --use_mixed_precision flag (default False = compare both modes)

                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    print(f"  ✅ Success: Results saved to {output_file}")
                    if memory_profile:
                        # Determine actual snapshot directory
                        snapshot_dir = memory_profile_dir if memory_profile_dir else str(results_path)
                        from glob import glob

                        pattern = (
                            f"memory_snapshot_*_{run_mode}_*ctx{context_length}_step{memory_profile_step}_*.pickle"
                        )
                        matched = sorted(glob(str(Path(snapshot_dir) / pattern)))
                        if matched:
                            print(f"  🧠 Memory snapshots ({len(matched)}):")
                            for p in matched[-2:]:  # show last couple likely from this run
                                print(f"     - {p}")
                        else:
                            print(
                                "  (No memory snapshot files found yet — ensure benchmark ran the profiled step or check directory.)"
                            )
                    successful += 1
                else:
                    print(f"  ❌ Failed with return code {result.returncode}")
                    print(f"  STDERR: {result.stderr}")
                    failed += 1

            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                failed += 1
                continue

    print("=" * 60)
    print("\nBenchmark sweep completed!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Results directory: {results_path}")

    if successful > 0:
        print("\nNext steps:")
        print(f"  1. Analyze results: uv run cs336_systems/analyze_results.py --results_dir {results_dir}")
        print("  2. Check generated tables in analysis/ directory")


if __name__ == "__main__":
    main()
