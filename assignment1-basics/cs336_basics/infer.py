import click
import torch
from jaxtyping import Int, Float
from torch import Tensor
from torch import nn
from einops import rearrange
from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.building_blocks import TransformerLM, load_checkpoint, softmax, AdamW


def create_padded_tensor(tokens_tensor: Int[Tensor, "1 seq_len"], context_length: int, device: torch.device):
    """
    Create a padded tensor from tokens, truncating if necessary.

    Args:
        tokens_tensor: Tensor of shape (1, seq_len) containing tokens
        context_length: Target length for padding/truncation
        device: Device to create tensors on

    Returns:
        Tensor of shape (1, context_length)
    """
    # Ensure tensor is on correct device
    tokens_tensor = tokens_tensor.to(device)
    seq_len = tokens_tensor.shape[1]

    if seq_len >= context_length:
        # Truncate if sequence is too long
        return tokens_tensor[:, :context_length]

    # Pad if sequence is too short
    padding_needed = context_length - seq_len
    padding = torch.zeros((1, padding_needed), dtype=tokens_tensor.dtype, device=device)
    return torch.cat([tokens_tensor, padding], dim=1)


def sample_next_token(logits: Float[Tensor, "1 vocab_size"], temperature: float, top_p_sampling_value: float):
    """
    Sample the next token using top-p sampling.

    Args:
        logits: Logits tensor of shape (1, vocab_size)
        temperature: Temperature for sampling
        top_p_sampling_value: Top-p value for nucleus sampling

    Returns:
        Sampled token tensor of shape (1, 1)
    """
    # Handle temperature = 0 (greedy sampling)
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Validate top_p_sampling_value
    top_p_sampling_value = max(0.0, min(1.0, top_p_sampling_value))

    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, dim=-1)

    # Top-p sampling
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff point - handle 2D tensor properly
    cutoff = torch.searchsorted(cumulative_probs.squeeze(0), top_p_sampling_value, right=True)
    cutoff = max(1, cutoff.item())  # Ensure at least one token is selected

    filtered_indices = sorted_indices[:, :cutoff]
    filtered_probs = sorted_probs[:, :cutoff]

    # Renormalize probabilities
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_index = torch.multinomial(filtered_probs, num_samples=1)
    return filtered_indices.gather(1, sampled_index)


def generate_tokens(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    context_length: int,
    top_p_sampling_value: float,
    device: torch.device,
):
    """
    Generate tokens using the model with the given prompt.

    Args:
        model: The transformer model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input prompt string
        max_output_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling
        context_length: Context length for the model
        top_p_sampling_value: Top-p value for nucleus sampling
        device: Device to run inference on

    Returns:
        List of generated tokens
    """
    # Encode initial prompt
    input_tokens = tokenizer.encode(prompt)
    current_tokens = rearrange(torch.tensor(input_tokens, dtype=torch.int32, device=device), " seq_len -> 1 seq_len")

    generated_tokens = []

    for _ in range(max_output_tokens):
        # Check if we've reached context length
        if current_tokens.shape[1] >= context_length:
            break

        # Create padded input tensor
        input_tensor = create_padded_tensor(current_tokens, context_length, device)

        # Get logits from model
        logits = model(input_tensor)

        # Get logits for the last non-padded position
        last_token_logits = logits[:, current_tokens.shape[1] - 1, :]

        # Sample next token
        next_token = sample_next_token(last_token_logits, temperature, top_p_sampling_value)

        # Add to generated tokens
        generated_tokens.append(next_token.item())

        # Update current tokens
        current_tokens = torch.cat([current_tokens, next_token], dim=1)

        # Check for end of text token (if applicable)
        eos_tokens = tokenizer.encode("<|endoftext|>")
        if eos_tokens and next_token.item() == eos_tokens[0]:
            break

    return generated_tokens


@click.command()
@click.option("--prompt", type=str, required=True, help="Prompt to decode.", prompt="input prompt")
@click.option(
    "--max_output_tokens",
    type=int,
    default=50,
    help="Maximum number of output tokens to generate.",
    prompt="max_output_tokens",
)
@click.option(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the checkpoint file.",
    default="./checkpoints/checkpoint_40000.pt",
)
@click.option(
    "--vocab_path",
    type=str,
    required=True,
    help="Path to the vocabulary file.",
    default="./data/bpe_vocab_TinyStoriesV2-GPT4-train.txt.pkl",
)
@click.option(
    "--merges_path",
    type=str,
    required=True,
    help="Path to the merges file.",
    default="./data/bpe_merges_TinyStoriesV2-GPT4-train.txt.pkl",
)
@click.option("--temperature", type=float, required=True, help="Temperature for sampling.", prompt="temperature")
@click.option(
    "--top_p_sampling_value", type=float, required=True, help="Top-p sampling value.", prompt="top_p_sampling_value"
)
@click.option("--device", type=str, default="cuda", help="Device to load the model on (e.g., 'cpu' or 'cuda').")
def infer(
    prompt: str,
    max_output_tokens: int,
    checkpoint: str,
    vocab_path: str,
    merges_path: str,
    temperature: float,
    top_p_sampling_value: float,
    device: str = "cpu",
):
    """
    Decode a checkpoint and print the model architecture.
    """
    device = torch.device(device)

    context_length = 256  # Adjust based on your model's context length

    # Load the model
    model = TransformerLM(
        vocab_size=10_000,
        d_model=512,  # You can adjust this based on your model's architecture
        context_length=context_length,  # Adjust based on your model's context length
        num_layers=4,  # Adjust based on your model's architecture
        num_heads=16,  # Adjust based on your model's architecture
        device=device,
        rope_theta=10_000,  # Adjust based on your model's architecture
        d_ff=1344,  # Adjust based on your model's architecture
    ).to(device)

    load_checkpoint(
        src=checkpoint,
        model=model,
        optimizer=AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2),
    )

    print(f"Model architecture:\n{model}")

    special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens
    )

    # Generate tokens
    generated_tokens = generate_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        context_length=context_length,
        top_p_sampling_value=top_p_sampling_value,
        device=device,
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")

    # Also print the full response (prompt + generated)
    full_tokens = tokenizer.encode(prompt) + generated_tokens
    full_text = tokenizer.decode(full_tokens)
    print(f"Full response: {full_text}")


if __name__ == "__main__":
    infer()
