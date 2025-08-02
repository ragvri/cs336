import os
from pathlib import Path
import time

import click
import numpy as np
import torch
import wandb
from tqdm import tqdm

from cs336_basics.building_blocks import (
    AdamW,
    TransformerLM,
    cosine_annealing_lr_schedule,
    cross_entropy_loss,
    gradient_clipping,
    load_data,
    save_checkpoint,
)


@click.command()
@click.option("--dataset", type=str, prompt="Dataset path", help="Path to the dataset file.")
@click.option("--batch_size", type=int, default=32, prompt="Batch size", help="Batch size for training.")
@click.option("--vocab_size", type=int, default=10000, prompt="Vocab size", help="Size of the vocabulary.")
@click.option("--context_length", type=int, default=1024, prompt="Context length", help="Context length for the model.")
@click.option("--d_model", type=int, default=512, prompt="Model dimension", help="Dimension of the model")
@click.option("--num_layers", type=int, default=6, prompt="Number of layers", help="Number of transformer layers.")
@click.option("--num_heads", type=int, default=8, prompt="Number of heads", help="Number of attention heads.")
@click.option(
    "--d_ff",
    type=int,
    default=None,
    help="Hidden dimension for the feed-forward network. If None, defaults to 8/3 * d_model.",
)
@click.option(
    "--rope_theta",
    type=float,
    default=10_000,
    help="Constant for the RoPE. If None, RoPE is disabled.",
    prompt="RoPE theta",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    prompt="Device (cpu/cuda)",
    help="Device to run the training on (e.g., 'cpu' or 'cuda').",
)
@click.option("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
@click.option("--iterations", type=int, default=None, help="Number of training iterations.")
@click.option(
    "--total_tokens_processed",
    type=int,
    default=None,
    help="bsz x total iterations x context_length",
    prompt="Total tokens processed",
)
@click.option(
    "--wandb_project",
    type=str,
    prompt="Wandb project name",
    default="cs336-assignment1",
    help="Wandb project name. If None, wandb logging is disabled.",
)
@click.option(
    "--wandb_run_name",
    type=str,
    prompt="Wandb run name",  # Added prompt for run name
    default=None,
    help="Wandb run name.",
)
def train(
    dataset: str,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int | None = None,
    rope_theta: float | None = None,
    device: str = "cpu",
    output_dir: str = "checkpoints",
    iterations: int | None = None,
    total_tokens_processed: int | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):
    """
    Train a Transformer Language Model on the given dataset.
    """
    # Calculate iterations from total_tokens_processed if needed
    if iterations is None and total_tokens_processed is None:
        iterations = 1000  # Default value
    elif iterations is None and total_tokens_processed is not None:
        iterations = total_tokens_processed // (batch_size * context_length)
        if iterations == 0:
            raise ValueError("total_tokens_processed is too small for the given batch_size and context_length")
    elif iterations is not None and total_tokens_processed is not None:
        raise ValueError("Cannot specify both iterations and total_tokens_processed. Please specify only one.")

    device = torch.device(device)

    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

    # Initialize wandb if project is specified
    if wandb_project:
        api_key = os.getenv("WANDB_API_KEY")
        # Check for API key in environment
        if api_key:
            wandb.login(key=api_key)

        wandb.init(
            entity="ragvri-student",  # Replace with your wandb entity
            project=wandb_project,
            name=wandb_run_name,
            config={
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "rope_theta": rope_theta,
                "device": str(device),
                "iterations": iterations,
            },
        )

    # Load dataset
    x = np.memmap(dataset, dtype=np.uint16, mode="r")
    # Initialize model and optimizer
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        context_length=context_length,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
    ).to(device)

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model}")
    print(f"Total parameters: {total_params:,}")
    print(f"Learnable parameters: {learnable_params:,}")

    progress_bar = tqdm(range(iterations), desc="Training")
    for iteration in progress_bar:
        # Load data
        data_start = time.time()
        input_tensor, target_tensor = load_data(x, batch_size, context_length, device)
        data_time = time.time() - data_start

        # Forward pass
        forward_start = time.time()
        logits = model(input_tensor)
        # Compute loss
        loss = cross_entropy_loss(logits.view(-1, vocab_size), target_tensor.view(-1))
        forward_time = time.time() - forward_start

        lr = cosine_annealing_lr_schedule(
            t=iteration,
            alpha_max=1e-1,
            alpha_min=1e-5,
            warmup_steps=100,
            cosine_annealing_steps=iterations,
        )
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)

        # Backward pass
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_norm=4.0)
        optimizer.step()
        backward_time = time.time() - backward_start

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{lr:.2e}',
            'data_ms': f'{data_time*1000:.1f}',
            'fwd_ms': f'{forward_time*1000:.1f}',
            'bwd_ms': f'{backward_time*1000:.1f}'
        })

        # Log to wandb if enabled
        if wandb_project:
            wandb.log(
                {
                    "iteration": iteration + 1,
                    "loss": loss.item(),
                    "learning_rate": lr,
                    "data_time_ms": data_time * 1000,
                    "forward_time_ms": forward_time * 1000,
                    "backward_time_ms": backward_time * 1000,
                }
            )

        # Save checkpoint every 100 iterations
        if (iteration + 1) % 100 == 0:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration + 1}.pt")
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)

    # Finish wandb run
    if wandb_project:
        wandb.finish()


if __name__ == "__main__":
    train()
