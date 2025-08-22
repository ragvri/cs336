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

logging.basicConfig(level=logging.INFO)


def benchmark(d_model: int, d_seq: int) -> dict:
    """
    benchmark attention for given head dimension output and sequence length
    Returns dict with results or OOM status
    """
    steps = warmup_steps = 100
    batch_size = 8

    try:
        # Create Q, K, V tensors once (outside the loops)
        Q = torch.randn(batch_size, d_seq, d_model, requires_grad=True, device="cuda")
        K = torch.randn(batch_size, d_seq, d_model, requires_grad=True, device="cuda")
        V = torch.randn(batch_size, d_seq, d_model, requires_grad=True, device="cuda")

        # Create mask once (reused for all iterations)
        seq_idx = torch.arange(d_seq, device="cuda")
        mask = seq_idx.unsqueeze(0) >= seq_idx.unsqueeze(1)

        logging.info("warmup up")
        for _ in range(warmup_steps):
            # Clear gradients from previous iteration
            if Q.grad is not None:
                Q.grad.zero_()
                K.grad.zero_()
                V.grad.zero_()

            # compute attention - forward (reusing same tensors)
            scaled_dot_product_attention(Q, K, V, mask=mask)
        logging.info("warmup done")

        forward_times = []
        backward_times = []
        memory_used_before_backward_passes = []

        for _ in range(steps):
            # Clear gradients from previous iteration
            if Q.grad is not None:
                Q.grad.zero_()
                K.grad.zero_()
                V.grad.zero_()

            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            # compute attention - forward (reusing same tensors)
            logits = scaled_dot_product_attention(Q, K, V, mask=mask)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            forward_times.append(end_time - start_time)

            # get the memory in use before backward pass starts
            memory_used = torch.cuda.memory_allocated()
            memory_used_before_backward_passes.append(memory_used)

            # start backward
            loss = logits.mean()
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            backward_times.append(end_time - start_time)

        avg_forward_time = sum(forward_times) / steps
        avg_backward_time = sum(backward_times) / steps
        avg_memory_used = sum(memory_used_before_backward_passes) / steps

        logging.info(f"Average forward time: {avg_forward_time:.4f} seconds")
        logging.info(f"Average backward time: {avg_backward_time:.4f} seconds")
        logging.info(
            f"Average memory used: {avg_memory_used:.0f} bytes ({avg_memory_used / 1024 / 1024 / 1024:.3f} GB)"
        )

        return {
            "forward_time": avg_forward_time,
            "backward_time": avg_backward_time,
            "memory_gb": avg_memory_used / 1024 / 1024 / 1024,
            "oom": False,
            "error": None,
        }

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA OOM for d_model={d_model}, d_seq={d_seq}: {e}")
        logging.info("Clearing CUDA cache and continuing...")
        torch.cuda.empty_cache()
        return {"forward_time": None, "backward_time": None, "memory_gb": None, "oom": True, "error": str(e)}
    except Exception as e:
        logging.error(f"Error benchmarking d_model={d_model}, d_seq={d_seq}: {e}")
        torch.cuda.empty_cache()
        return {"forward_time": None, "backward_time": None, "memory_gb": None, "oom": False, "error": str(e)}


def benchmark_different_dimensions() -> dict:
    """Run benchmarks across all parameter combinations and return results"""
    d_model_list = [16, 32, 64, 128]
    d_seq_list = [256, 1024, 4096, 8192, 16384]

    results = {}
    total_configs = len(d_model_list) * len(d_seq_list)
    current_config = 0

    for d_model in d_model_list:
        for d_seq in d_seq_list:
            current_config += 1
            logging.info(f"Benchmarking d_model={d_model}, d_seq={d_seq} ({current_config}/{total_configs})")

            result = benchmark(d_model, d_seq)
            results[(d_model, d_seq)] = result

            logging.info("Benchmarking complete\n")

    return results


def save_results(results: dict, filename: str = None):
    """Save results to JSON file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_benchmark_results_{timestamp}.json"

    # Convert tuple keys to strings for JSON serialization
    json_results = {f"{d_model}_{d_seq}": result for (d_model, d_seq), result in results.items()}

    with open(filename, "w") as f:
        json.dump(json_results, f, indent=2)

    logging.info(f"Results saved to {filename}")
    return filename


def plot_results(results: dict, save_plots: bool = True):
    """Create heatmap visualizations of the benchmark results"""
    d_model_list = [16, 32, 64, 128]
    d_seq_list = [256, 1024, 4096, 8192, 16384]

    # Create matrices for each metric
    forward_matrix = np.full((len(d_model_list), len(d_seq_list)), np.nan)
    backward_matrix = np.full((len(d_model_list), len(d_seq_list)), np.nan)
    memory_matrix = np.full((len(d_model_list), len(d_seq_list)), np.nan)

    # Fill matrices with results
    for i, d_model in enumerate(d_model_list):
        for j, d_seq in enumerate(d_seq_list):
            result = results.get((d_model, d_seq), {})
            if not result.get("oom", True) and result.get("error") is None:
                forward_matrix[i, j] = result["forward_time"]
                backward_matrix[i, j] = result["backward_time"]
                memory_matrix[i, j] = result["memory_gb"]

    # Create the plot with better proportions
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Attention Scaling Benchmark Results", fontsize=16, fontweight="bold", y=0.98)

    # Define common parameters  
    d_seq_labels = [f"{seq // 1000}K" if seq >= 1000 else str(seq) for seq in d_seq_list]

    # Create better colormaps with more contrast
    from matplotlib.colors import ListedColormap
    
    # Use a more contrasting colormap - plasma has better visibility
    plasma = plt.cm.get_cmap('plasma')
    colors = plasma(np.linspace(0, 1, 256))
    cmap_with_nan = ListedColormap(colors)
    cmap_with_nan.set_bad(color='#d3d3d3')  # Light gray for NaN

    # Plot 1: Forward Time
    # Get data range and use better color scaling
    forward_valid = forward_matrix[~np.isnan(forward_matrix)]
    if len(forward_valid) > 0:
        # Use percentiles for better color distribution
        vmin_f = np.percentile(forward_valid, 5)
        vmax_f = np.percentile(forward_valid, 95)
    else:
        vmin_f, vmax_f = 0, 1
        
    im1 = axes[0].imshow(forward_matrix, cmap=cmap_with_nan, vmin=vmin_f, vmax=vmax_f, 
                         aspect='auto', interpolation='nearest')
    axes[0].set_title("Forward Pass Time (seconds)", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Sequence Length", fontsize=10)
    axes[0].set_ylabel("d_model", fontsize=10)
    axes[0].set_xticks(range(len(d_seq_list)))
    axes[0].set_xticklabels(d_seq_labels, fontsize=9)
    axes[0].set_yticks(range(len(d_model_list)))
    axes[0].set_yticklabels(d_model_list, fontsize=9)

    # Add value annotations for forward time
    for i in range(len(d_model_list)):
        for j in range(len(d_seq_list)):
            if not np.isnan(forward_matrix[i, j]):
                text = f"{forward_matrix[i, j]:.4f}"
                # Use better contrast - always white text with black outline
                axes[0].text(j, i, text, ha="center", va="center", 
                           color="white", fontsize=8, fontweight='bold',
                           path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
            else:
                # Show OOM for missing data
                result = results.get((d_model_list[i], d_seq_list[j]), {})
                if result.get("oom", False):
                    axes[0].text(j, i, "OOM", ha="center", va="center", 
                               color="darkred", fontsize=8, fontweight='bold',
                               path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])

    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.ax.tick_params(labelsize=8)

    # Plot 2: Backward Time
    backward_valid = backward_matrix[~np.isnan(backward_matrix)]
    if len(backward_valid) > 0:
        vmin_b = np.percentile(backward_valid, 5)
        vmax_b = np.percentile(backward_valid, 95)
    else:
        vmin_b, vmax_b = 0, 1
        
    im2 = axes[1].imshow(backward_matrix, cmap=cmap_with_nan, vmin=vmin_b, vmax=vmax_b,
                         aspect='auto', interpolation='nearest')
    axes[1].set_title("Backward Pass Time (seconds)", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Sequence Length", fontsize=10)
    axes[1].set_ylabel("d_model", fontsize=10)
    axes[1].set_xticks(range(len(d_seq_list)))
    axes[1].set_xticklabels(d_seq_labels, fontsize=9)
    axes[1].set_yticks(range(len(d_model_list)))
    axes[1].set_yticklabels(d_model_list, fontsize=9)

    # Add value annotations for backward time
    for i in range(len(d_model_list)):
        for j in range(len(d_seq_list)):
            if not np.isnan(backward_matrix[i, j]):
                text = f"{backward_matrix[i, j]:.4f}"
                axes[1].text(j, i, text, ha="center", va="center",
                           color="white", fontsize=8, fontweight='bold',
                           path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
            else:
                result = results.get((d_model_list[i], d_seq_list[j]), {})
                if result.get("oom", False):
                    axes[1].text(j, i, "OOM", ha="center", va="center", 
                               color="darkred", fontsize=8, fontweight='bold',
                               path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])

    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.ax.tick_params(labelsize=8)

    # Plot 3: Memory Usage
    memory_valid = memory_matrix[~np.isnan(memory_matrix)]
    if len(memory_valid) > 0:
        vmin_m = np.percentile(memory_valid, 5)
        vmax_m = np.percentile(memory_valid, 95)
    else:
        vmin_m, vmax_m = 0, 1
        
    im3 = axes[2].imshow(memory_matrix, cmap=cmap_with_nan, vmin=vmin_m, vmax=vmax_m,
                         aspect='auto', interpolation='nearest')
    axes[2].set_title("Memory Usage (GB)", fontweight="bold", fontsize=12)
    axes[2].set_xlabel("Sequence Length", fontsize=10)
    axes[2].set_ylabel("d_model", fontsize=10)
    axes[2].set_xticks(range(len(d_seq_list)))
    axes[2].set_xticklabels(d_seq_labels, fontsize=9)
    axes[2].set_yticks(range(len(d_model_list)))
    axes[2].set_yticklabels(d_model_list, fontsize=9)

    # Add value annotations for memory
    for i in range(len(d_model_list)):
        for j in range(len(d_seq_list)):
            if not np.isnan(memory_matrix[i, j]):
                text = f"{memory_matrix[i, j]:.2f}"
                axes[2].text(j, i, text, ha="center", va="center",
                           color="white", fontsize=8, fontweight='bold',
                           path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')])
            else:
                result = results.get((d_model_list[i], d_seq_list[j]), {})
                if result.get("oom", False):
                    axes[2].text(j, i, "OOM", ha="center", va="center", 
                               color="darkred", fontsize=8, fontweight='bold',
                               path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])

    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.ax.tick_params(labelsize=8)

    # Remove grid and improve layout
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=9)
        # Add subtle grid lines
        ax.set_xticks(np.arange(-0.5, len(d_seq_list), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(d_model_list), 1), minor=True) 
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"attention_benchmark_plots_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        logging.info(f"Plots saved to {plot_filename}")

    # plt.show()  # Commented out for SSH environments

    # Print summary
    logging.info("\n" + "=" * 60)
    logging.info("BENCHMARK SUMMARY")
    logging.info("=" * 60)

    successful_configs = sum(
        1 for result in results.values() if not result.get("oom", True) and result.get("error") is None
    )
    oom_configs = sum(1 for result in results.values() if result.get("oom", False))
    error_configs = sum(
        1 for result in results.values() if result.get("error") is not None and not result.get("oom", False)
    )

    logging.info(f"Total configurations: {len(results)}")
    logging.info(f"Successful: {successful_configs}")
    logging.info(f"OOM errors: {oom_configs}")
    logging.info(f"Other errors: {error_configs}")

    if successful_configs > 0:
        successful_results = [r for r in results.values() if not r.get("oom", True) and r.get("error") is None]
        forward_times = [r["forward_time"] for r in successful_results]
        backward_times = [r["backward_time"] for r in successful_results]
        memory_usage = [r["memory_gb"] for r in successful_results]

        logging.info(f"\nForward time range: {min(forward_times):.4f}s - {max(forward_times):.4f}s")
        logging.info(f"Backward time range: {min(backward_times):.4f}s - {max(backward_times):.4f}s")
        logging.info(f"Memory usage range: {min(memory_usage):.3f}GB - {max(memory_usage):.3f}GB")


if __name__ == "__main__":
    # Run benchmarks
    results = benchmark_different_dimensions()

    # Save results
    save_results(results)

    # Create plots
    plot_results(results)
