import json
import logging
import timeit
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for SSH environments
import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
import torch
from cs336_basics.model import scaled_dot_product_attention

torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO)


def benchmark_implementation(impl_func, Q, K, V, mask, steps=100, warmup_steps=100):
    """Benchmark a specific attention implementation"""

    # Warmup
    for _ in range(warmup_steps):
        if Q.grad is not None:
            Q.grad.zero_()
            K.grad.zero_()
            V.grad.zero_()
        impl_func(Q, K, V, mask)

    forward_times = []
    backward_times = []
    memory_used_before_backward_passes = []

    for _ in range(steps):
        if Q.grad is not None:
            Q.grad.zero_()
            K.grad.zero_()
            V.grad.zero_()

        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        logits = impl_func(Q, K, V, mask)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        forward_times.append(end_time - start_time)

        memory_used = torch.cuda.memory_allocated()
        memory_used_before_backward_passes.append(memory_used)

        loss = logits.mean()
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        backward_times.append(end_time - start_time)

    return {
        "forward_time": sum(forward_times) / steps,
        "backward_time": sum(backward_times) / steps,
        "memory_gb": sum(memory_used_before_backward_passes) / steps / 1024 / 1024 / 1024,
    }


def benchmark_comparison(d_model: int, d_seq: int) -> dict:
    """
    Compare custom attention implementation vs torch.compiled version
    Returns dict with results for both implementations
    """
    batch_size = 8

    try:
        # Create tensors
        Q = torch.randn(batch_size, d_seq, d_model, requires_grad=True, device="cuda")
        K = torch.randn(batch_size, d_seq, d_model, requires_grad=True, device="cuda")
        V = torch.randn(batch_size, d_seq, d_model, requires_grad=True, device="cuda")

        seq_idx = torch.arange(d_seq, device="cuda")
        mask = seq_idx.unsqueeze(0) >= seq_idx.unsqueeze(1)

        results = {}

        # 1. Original custom implementation
        logging.info("Benchmarking original custom implementation...")
        results["original"] = benchmark_implementation(scaled_dot_product_attention, Q, K, V, mask)
        results["original"]["oom"] = False
        results["original"]["error"] = None

        # 2. torch.compiled custom implementation
        logging.info("Benchmarking torch.compiled custom implementation...")
        compiled_attention = torch.compile(scaled_dot_product_attention)
        results["compiled"] = benchmark_implementation(compiled_attention, Q, K, V, mask)
        results["compiled"]["oom"] = False
        results["compiled"]["error"] = None

        return results

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA OOM for d_model={d_model}, d_seq={d_seq}: {e}")
        torch.cuda.empty_cache()
        return {
            "original": {"forward_time": None, "backward_time": None, "memory_gb": None, "oom": True, "error": str(e)},
            "compiled": {"forward_time": None, "backward_time": None, "memory_gb": None, "oom": True, "error": str(e)},
        }
    except Exception as e:
        logging.error(f"Error benchmarking d_model={d_model}, d_seq={d_seq}: {e}")
        torch.cuda.empty_cache()
        return {
            "original": {"forward_time": None, "backward_time": None, "memory_gb": None, "oom": False, "error": str(e)},
            "compiled": {"forward_time": None, "backward_time": None, "memory_gb": None, "oom": False, "error": str(e)},
        }


def benchmark_different_dimensions() -> dict:
    """Run benchmarks across all parameter combinations"""
    d_model_list = [16, 32, 64, 128]
    d_seq_list = [256, 1024, 4096, 8192, 16384]

    results = {}
    total_configs = len(d_model_list) * len(d_seq_list)
    current_config = 0

    for d_model in d_model_list:
        for d_seq in d_seq_list:
            current_config += 1
            logging.info(f"Benchmarking d_model={d_model}, d_seq={d_seq} ({current_config}/{total_configs})")

            result = benchmark_comparison(d_model, d_seq)
            results[(d_model, d_seq)] = result

            logging.info("Benchmarking complete\n")

    return results


def save_results(results: dict, filename: str = None):
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"compiled_attention_comparison_{timestamp}.json"

    # Convert tuple keys to strings for JSON serialization
    json_results = {}
    for (d_model, d_seq), impl_results in results.items():
        json_results[f"{d_model}_{d_seq}"] = impl_results

    with open(filename, "w") as f:
        json.dump(json_results, f, indent=2)

    logging.info(f"Results saved to {filename}")
    return filename


def plot_comparison_results(results: dict, save_plots: bool = True):
    """Create comparison visualizations"""
    d_model_list = [16, 32, 64, 128]
    d_seq_list = [256, 1024, 4096, 8192, 16384]

    # Create subplots for comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Custom Attention: Original vs torch.compile", fontsize=16, fontweight="bold")

    d_seq_labels = [f"{seq // 1000}K" if seq >= 1000 else str(seq) for seq in d_seq_list]

    from matplotlib.colors import ListedColormap

    plasma = plt.cm.get_cmap("plasma")
    colors = plasma(np.linspace(0, 1, 256))
    cmap_with_nan = ListedColormap(colors)
    cmap_with_nan.set_bad(color="#d3d3d3")

    # Plot configurations: (implementation, metric, title)
    plot_configs = [
        ("original", "forward_time", "Original - Forward Time (s)"),
        ("compiled", "forward_time", "Compiled - Forward Time (s)"),
        ("original", "backward_time", "Original - Backward Time (s)"),
        ("compiled", "backward_time", "Compiled - Backward Time (s)"),
        ("original", "memory_gb", "Original - Memory Usage (GB)"),
        ("compiled", "memory_gb", "Compiled - Memory Usage (GB)"),
    ]

    for idx, (impl, metric, title) in enumerate(plot_configs):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        # Create matrix for this implementation and metric
        matrix = np.full((len(d_model_list), len(d_seq_list)), np.nan)

        for i, d_model in enumerate(d_model_list):
            for j, d_seq in enumerate(d_seq_list):
                result = results.get((d_model, d_seq), {}).get(impl, {})
                if not result.get("oom", True) and result.get("error") is None:
                    matrix[i, j] = result[metric]

        # Plot heatmap
        valid_data = matrix[~np.isnan(matrix)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 5)
            vmax = np.percentile(valid_data, 95)
        else:
            vmin, vmax = 0, 1

        im = ax.imshow(matrix, cmap=cmap_with_nan, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Sequence Length", fontsize=9)
        ax.set_ylabel("d_model", fontsize=9)
        ax.set_xticks(range(len(d_seq_list)))
        ax.set_xticklabels(d_seq_labels, fontsize=8)
        ax.set_yticks(range(len(d_model_list)))
        ax.set_yticklabels(d_model_list, fontsize=8)

        # Add value annotations
        for i in range(len(d_model_list)):
            for j in range(len(d_seq_list)):
                if not np.isnan(matrix[i, j]):
                    if metric == "memory_gb":
                        text = f"{matrix[i, j]:.2f}"
                    else:
                        text = f"{matrix[i, j]:.4f}"
                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=7,
                        fontweight="bold",
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=1, foreground="black")],
                    )
                else:
                    result = results.get((d_model_list[i], d_seq_list[j]), {}).get(impl, {})
                    if result.get("oom", False):
                        ax.text(j, i, "OOM", ha="center", va="center", color="darkred", fontsize=7, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"compiled_attention_comparison_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        logging.info(f"Comparison plots saved to {plot_filename}")

    # Create speedup analysis
    create_speedup_analysis(results, save_plots)


def create_speedup_analysis(results: dict, save_plots: bool = True):
    """Create detailed speedup analysis"""
    d_model_list = [16, 32, 64, 128]
    d_seq_list = [256, 1024, 4096, 8192, 16384]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("torch.compile Speedup Analysis", fontsize=14, fontweight="bold")

    metrics = [("forward_time", "Forward Pass"), ("backward_time", "Backward Pass")]

    for metric_idx, (metric, title) in enumerate(metrics):
        ax = axes[metric_idx]

        speedups = []
        configs = []
        original_times = []
        compiled_times = []

        for d_model in d_model_list:
            for d_seq in d_seq_list:
                result = results.get((d_model, d_seq), {})
                orig_result = result.get("original", {})
                comp_result = result.get("compiled", {})

                orig_time = orig_result.get(metric)
                comp_time = comp_result.get(metric)

                if (
                    orig_time
                    and comp_time
                    and not orig_result.get("oom", True)
                    and not comp_result.get("oom", True)
                    and orig_result.get("error") is None
                    and comp_result.get("error") is None
                ):
                    speedup = orig_time / comp_time
                    speedups.append(speedup)
                    configs.append(f"{d_model}x{d_seq}")
                    original_times.append(orig_time)
                    compiled_times.append(comp_time)

        # Bar plot of speedups
        x = np.arange(len(configs))
        bars = ax.bar(x, speedups, alpha=0.7, color=["green" if s > 1 else "red" for s in speedups])

        ax.set_xlabel("Configuration (d_model x d_seq)")
        ax.set_ylabel("Speedup Factor (Original / Compiled)")
        ax.set_title(f"{title} Speedup")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right")
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.7, label="No speedup")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

        # Add value labels on bars
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        speedup_filename = f"compiled_attention_speedup_{timestamp}.png"
        plt.savefig(speedup_filename, dpi=300, bbox_inches="tight")
        logging.info(f"Speedup analysis saved to {speedup_filename}")


def print_summary(results: dict):
    """Print comprehensive summary of results"""
    logging.info("\n" + "=" * 80)
    logging.info("TORCH.COMPILE ATTENTION COMPARISON SUMMARY")
    logging.info("=" * 80)

    implementations = ["original", "compiled"]
    impl_labels = ["Original", "torch.compile"]

    total_configs = len(results)

    for impl, label in zip(implementations, impl_labels):
        successful = sum(
            1 for r in results.values() if not r.get(impl, {}).get("oom", True) and r.get(impl, {}).get("error") is None
        )
        oom = sum(1 for r in results.values() if r.get(impl, {}).get("oom", False))
        errors = sum(
            1
            for r in results.values()
            if r.get(impl, {}).get("error") is not None and not r.get(impl, {}).get("oom", False)
        )

        logging.info(f"\n{label}:")
        logging.info(f"  Successful: {successful}/{total_configs}")
        logging.info(f"  OOM: {oom}")
        logging.info(f"  Errors: {errors}")

        if successful > 0:
            successful_results = [
                r[impl]
                for r in results.values()
                if not r.get(impl, {}).get("oom", True) and r.get(impl, {}).get("error") is None
            ]
            forward_times = [r["forward_time"] for r in successful_results]
            backward_times = [r["backward_time"] for r in successful_results]
            memory_usage = [r["memory_gb"] for r in successful_results]

            logging.info(
                f"  Forward time: {min(forward_times):.4f}s - {max(forward_times):.4f}s (avg: {np.mean(forward_times):.4f}s)"
            )
            logging.info(
                f"  Backward time: {min(backward_times):.4f}s - {max(backward_times):.4f}s (avg: {np.mean(backward_times):.4f}s)"
            )
            logging.info(
                f"  Memory usage: {min(memory_usage):.3f}GB - {max(memory_usage):.3f}GB (avg: {np.mean(memory_usage):.3f}GB)"
            )

    # Calculate overall speedup statistics
    forward_speedups = []
    backward_speedups = []

    for result in results.values():
        orig = result.get("original", {})
        comp = result.get("compiled", {})

        if (
            not orig.get("oom", True)
            and not comp.get("oom", True)
            and orig.get("error") is None
            and comp.get("error") is None
        ):
            if orig.get("forward_time") and comp.get("forward_time"):
                forward_speedups.append(orig["forward_time"] / comp["forward_time"])
            if orig.get("backward_time") and comp.get("backward_time"):
                backward_speedups.append(orig["backward_time"] / comp["backward_time"])

    if forward_speedups:
        logging.info(f"\nForward Pass Speedups:")
        logging.info(f"  Min: {min(forward_speedups):.2f}x")
        logging.info(f"  Max: {max(forward_speedups):.2f}x")
        logging.info(f"  Average: {np.mean(forward_speedups):.2f}x")
        logging.info(f"  Median: {np.median(forward_speedups):.2f}x")

    if backward_speedups:
        logging.info(f"\nBackward Pass Speedups:")
        logging.info(f"  Min: {min(backward_speedups):.2f}x")
        logging.info(f"  Max: {max(backward_speedups):.2f}x")
        logging.info(f"  Average: {np.mean(backward_speedups):.2f}x")
        logging.info(f"  Median: {np.median(backward_speedups):.2f}x")


if __name__ == "__main__":
    # Run comparison benchmarks
    results = benchmark_different_dimensions()

    # Save results
    save_results(results)

    # Create plots
    plot_comparison_results(results)

    # Print summary
    print_summary(results)
