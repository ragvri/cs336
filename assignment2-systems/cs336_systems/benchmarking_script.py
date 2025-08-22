import logging
import timeit
from contextlib import nullcontext
import statistics
import json
import os
from datetime import datetime
import uuid
from typing import Any

import click
import torch
import torch.cuda.nvtx as nvtx
import tqdm
from cs336_basics.model import BasicsTransformerLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark_with_precision(
    model: torch.nn.Module,
    data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    precision_mode: str,
    number_steps: int = 100,
    run_mode: str = "train",
) -> tuple[list[float], list[float], list[float]]:
    """Run benchmark with specified precision mode.

    run_mode: 'train' (forward + backward + optimizer) or 'forward' (inference only)
    """
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision_mode == "mixed" else nullcontext()
    )
    forward_times = []
    backward_times = []
    optimizer_times = []

    with autocast_ctx:
        for step in tqdm.trange(number_steps, desc=f"Benchmarking ({precision_mode} precision)"):
            nvtx.range_push(f"Step {step + 1}")

            # Forward pass timing
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            with nvtx.range("Forward pass"):
                logits = model(data)
                loss = logits.mean()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            forward_times.append(end_time - start_time)

            if run_mode == "train":
                # Backward pass timing
                torch.cuda.synchronize()
                start_time = timeit.default_timer()
                with nvtx.range("backward pass"):
                    loss.backward()
                torch.cuda.synchronize()
                end_time = timeit.default_timer()
                backward_times.append(end_time - start_time)

                # Optimizer step timing
                torch.cuda.synchronize()
                start_time = timeit.default_timer()
                with nvtx.range("optimizer step"):
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.synchronize()
                end_time = timeit.default_timer()
                optimizer_times.append(end_time - start_time)

            nvtx.range_pop()

    return forward_times, backward_times, optimizer_times


@click.command()
@click.option("--num_layers", default=12, help="Number of layers in the transformer model.")
@click.option("--num_heads", default=12, help="Number of attention heads in the transformer model.")
@click.option("--d_model", default=768, help="Dimension of the model.")
@click.option("--d_ff", default=3072, help="Dimension of the feedforward layer.")
@click.option("--batch_size", default=4, help="Batch size for benchmarking.")
@click.option("--context_length", default=128, help="Context length (sequence length).")
@click.option("--use_mixed_precision", is_flag=True, help="Use BF16 mixed precision.")
@click.option(
    "--precision",
    type=click.Choice(["both", "full", "mixed"]),
    default="both",
    help="Which precision modes to benchmark: both (full + mixed), full only, or mixed only. Overrides --use_mixed_precision if set.",
)
@click.option("--output_file", default=None, help="File to save structured results (JSON).")
@click.option("--warmup_steps", default=50, help="Number of warmup steps.")
@click.option("--benchmark_steps", default=100, help="Number of benchmark steps.")
@click.option(
    "--run_mode",
    type=click.Choice(["forward", "train"]),
    default="train",
    help="Run only the forward pass (inference) or a full training step (forward+backward+optimizer).",
)
@click.option("--memory_profile", is_flag=True, help="Enable detailed CUDA memory allocation profiling (slow).")
@click.option(
    "--memory_profile_dir",
    default=None,
    help="Directory to store memory snapshot pickle files (defaults to output file directory or CWD).",
)
@click.option(
    "--memory_profile_step",
    default=1,
    show_default=True,
    help="Which (1-indexed) benchmark step to capture a memory snapshot for when memory profiling is enabled. Other steps run without memory tracing.",
)
def benchmark_model(
    num_layers: int,
    num_heads: int,
    d_model: int,
    d_ff: int,
    batch_size: int,
    context_length: int,
    use_mixed_precision: bool,
    precision: str,
    output_file: str | None,
    warmup_steps: int,
    benchmark_steps: int,
    run_mode: str,
    memory_profile: bool,
    memory_profile_dir: str | None,
    memory_profile_step: int,
) -> dict[str, Any]:
    vocab_size = 10_000
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    logger.info(f"Benchmarking model with {num_layers} layers, {num_heads} heads, {d_model} d_model, {d_ff} d_ff")
    logger.info(f"Batch size: {batch_size}, Context length: {context_length}")
    logger.info(f"Run ID: {run_id}")

    with nvtx.range("define the model"):
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=0.1,
        )
        if torch.cuda.is_available():
            logger.info("Using CUDA for benchmarking.")
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    data = torch.randint(0, vocab_size, (batch_size, context_length))
    data = data.cuda()

    # Warmup
    with nvtx.range("warmup"):
        logger.info(f"Warmup steps: {warmup_steps}")
        warmup_start = timeit.default_timer()
        for _ in tqdm.trange(warmup_steps, desc="Warmup"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_mixed_precision else nullcontext():
                model(data)
        warmup_end = timeit.default_timer()
        warmup_time = warmup_end - warmup_start
        logger.info(f"Warmup complete in {warmup_time:.4f} seconds. Starting benchmarking...")

    number_steps = benchmark_steps
    if memory_profile and memory_profile_step > number_steps:
        logger.warning(
            f"Requested memory_profile_step={memory_profile_step} exceeds number of steps {number_steps}; defaulting to last step."
        )
        memory_profile_step = number_steps

    # Run benchmarks for both precision modes if not specified, otherwise just the requested one
    # Determine precision modes
    if precision != "both":
        precision_modes = [precision]
    else:
        # Legacy behavior: if --use_mixed_precision given alone, run only mixed
        if use_mixed_precision:
            precision_modes = ["mixed"]
        else:
            precision_modes = ["full", "mixed"]

    results = {}
    # Resolve memory profile output directory
    if memory_profile:
        if memory_profile_dir is None:
            # If an output_file is provided, reuse its directory; else current working directory
            if output_file:
                memory_profile_dir = os.path.dirname(output_file)
            else:
                memory_profile_dir = os.getcwd()
        os.makedirs(memory_profile_dir, exist_ok=True)
        logger.info(f"Memory profiling enabled. Snapshots will be written to: {memory_profile_dir}")

    for precision_mode in precision_modes:
        logger.info(f"Running benchmark with {precision_mode} precision...")
        # Reset optimizer state
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        forward_times = []
        backward_times = []
        optimizer_times = []

        # If memory profiling, we will wrap only the target step with recording
        for step_idx in range(number_steps):
            capture_this_step = memory_profile and (step_idx + 1) == memory_profile_step
            if capture_this_step:
                torch.cuda.memory._record_memory_history(max_entries=1_000_000)
                logger.info(
                    f"Memory profiling active for step {memory_profile_step} (precision={precision_mode}). Capturing snapshot."
                )
            ft, bt, ot = run_benchmark_with_precision(
                model,
                data,
                optimizer,
                precision_mode,
                number_steps=1,
                run_mode=run_mode,
            )
            forward_times.extend(ft)
            backward_times.extend(bt)
            optimizer_times.extend(ot)
            if capture_this_step:
                snapshot_filename = (
                    f"memory_snapshot_"
                    f"{precision_mode}_"
                    f"{run_mode}_"
                    f"{num_layers}L_{num_heads}H_{d_model}D_{d_ff}FF_"
                    f"bs{batch_size}_ctx{context_length}_"
                    f"step{memory_profile_step}_{run_id}.pickle"
                )
                snapshot_path = os.path.join(memory_profile_dir, snapshot_filename)
                torch.cuda.memory._dump_snapshot(snapshot_path)
                torch.cuda.memory._record_memory_history(enabled=None)
                logger.info(f"Memory snapshot written to {snapshot_path}")

        def _stats(lst):
            if not lst:
                return (None, None)
            if len(lst) == 1:
                return (lst[0], 0.0)
            return (statistics.mean(lst), statistics.stdev(lst))

        results[precision_mode] = {
            "forward": _stats(forward_times),
            "backward": _stats(backward_times),
            "optimizer": _stats(optimizer_times),
        }
    # (No per-mode snapshot here; handled per-step above.)

    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Prepare structured results
    structured_results = {
        "run_id": run_id,
        "timestamp": timestamp,
        "hyperparameters": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_model": d_model,
            "d_ff": d_ff,
            "batch_size": batch_size,
            "context_length": context_length,
            "vocab_size": vocab_size,
            "total_parameters": total_params,
        },
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available()
            else None,
        },
        "benchmark_config": {
            "warmup_steps": warmup_steps,
            "benchmark_steps": number_steps,
        },
        "results": {},
    }

    # Add results for each precision mode
    for precision_mode in precision_modes:
        forward_mean, forward_std = results[precision_mode]["forward"]
        backward_mean, backward_std = results[precision_mode]["backward"]
        optimizer_mean, optimizer_std = results[precision_mode]["optimizer"]

        total_time = None
        if forward_mean is not None:
            total_time = forward_mean
            if backward_mean is not None:
                total_time += backward_mean
            if optimizer_mean is not None:
                total_time += optimizer_mean

        structured_results["results"][precision_mode] = {
            "forward_pass": {"mean_time": forward_mean, "std_time": forward_std},
            "backward_pass": {"mean_time": backward_mean, "std_time": backward_std},
            "optimizer_step": {"mean_time": optimizer_mean, "std_time": optimizer_std},
            "total_step_time": {"mean_time": total_time},
        }

    # Add speedup comparison if both modes tested
    if len(precision_modes) == 2:
        full_forward = results["full"]["forward"][0]
        mixed_forward = results["mixed"]["forward"][0]
        backward_speedup = None
        if results["full"]["backward"][0] is not None and results["mixed"]["backward"][0] is not None:
            backward_speedup = results["full"]["backward"][0] / results["mixed"]["backward"][0]

        structured_results["speedup_analysis"] = {
            "forward_speedup": full_forward / mixed_forward if mixed_forward else None,
            "backward_speedup": backward_speedup,
        }

    # Print results
    print(f"\n=== Results for {num_layers}L-{num_heads}H-{d_model}D model ===")
    print(f"Total parameters: {total_params:,}")
    for precision_mode in precision_modes:
        print(f"\n{precision_mode.upper()} PRECISION:")
        forward_mean, forward_std = results[precision_mode]["forward"]
        backward_mean, backward_std = results[precision_mode]["backward"]
        optimizer_mean, optimizer_std = results[precision_mode]["optimizer"]

        def _fmt(label, mean, std):
            if mean is None:
                print(f"  {label:<13} - Skipped")
            else:
                print(f"  {label:<13} - Mean: {mean:.4f}s, Std: {std:.4f}s")

        _fmt("Forward pass", forward_mean, forward_std)
        _fmt("Backward pass", backward_mean, backward_std)
        _fmt("Optimizer", optimizer_mean, optimizer_std)

    # Compare if we have both
    if len(precision_modes) == 2:
        print("\n=== SPEEDUP COMPARISON ===")
        forward_speedup = structured_results["speedup_analysis"]["forward_speedup"]
        backward_speedup = structured_results["speedup_analysis"]["backward_speedup"]
        if forward_speedup:
            print(f"Forward pass speedup:  {forward_speedup:.2f}x ({'faster' if forward_speedup > 1 else 'slower'})")
        if backward_speedup:
            print(f"Backward pass speedup: {backward_speedup:.2f}x ({'faster' if backward_speedup > 1 else 'slower'})")

    # Save structured results to file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(structured_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return structured_results


if __name__ == "__main__":
    benchmark_model()
