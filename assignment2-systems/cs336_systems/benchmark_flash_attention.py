import torch
import triton
import triton.testing
from cs336_systems.flash_attention import FlashAttentionTriton
import pandas as pd
from einops import rearrange


# Define a naive PyTorch attention implementation (without Flash Attention)
def torch_attention(q, k, v, causal):
    """
    Naive attention implementation using standard PyTorch operations.
    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, d_head)
        k: Key tensor of shape (batch, n_heads, seq_len, d_head)
        v: Value tensor of shape (batch, n_heads, seq_len, d_head)
        causal: Whether to apply causal masking
    Returns:
        Output tensor of shape (batch, n_heads, seq_len, d_head)
    """
    # Compute attention scores
    scale = 1.0 / (q.size(-1) ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention weights to values
    output = torch.matmul(attn_weights, v)
    return output


def triton_attention(q, k, v, causal):
    # Reshape from (batch, n_heads, seq_len, d_head) to (batch, seq_len, n_heads * d_head)
    q_reshaped = rearrange(q, "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)")
    k_reshaped = rearrange(k, "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)")
    v_reshaped = rearrange(v, "batch n_heads seq_len d_head -> batch seq_len (n_heads d_head)")

    # Apply FlashAttentionTriton
    output_reshaped = FlashAttentionTriton.apply(q_reshaped, k_reshaped, v_reshaped, causal)

    # Reshape back to (batch, n_heads, seq_len, d_head)
    output = rearrange(
        output_reshaped, "batch seq_len (n_heads d_head) -> batch n_heads seq_len d_head", n_heads=q.shape[1]
    )
    return output


def benchmark(seq_len, d_head, precision, causal=True):
    device = "cuda"
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

    # Input tensors
    q = torch.randn((1, 1, seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((1, 1, seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((1, 1, seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn((1, 1, seq_len, d_head), dtype=dtype, device=device)

    # Quantiles for triton.testing.do_bench
    quantiles = [0.5, 0.2, 0.8]

    # --- PyTorch Benchmark ---
    try:
        # Forward pass
        pytorch_fwd_latency = triton.testing.do_bench(lambda: torch_attention(q, k, v, causal), quantiles=quantiles)[0]

        # Forward-Backward pass
        pytorch_fwdbwd_latency = triton.testing.do_bench(
            lambda: torch_attention(q, k, v, causal).backward(dout, retain_graph=True), quantiles=quantiles
        )[0]

        # Backward pass (approximated)
        pytorch_bwd_latency = pytorch_fwdbwd_latency - pytorch_fwd_latency
    except torch.cuda.OutOfMemoryError as e:
        print(f"PyTorch OOM for seq_len={seq_len}, d_head={d_head}, precision={precision}: {e}")
        pytorch_fwd_latency, pytorch_bwd_latency, pytorch_fwdbwd_latency = float("nan"), float("nan"), float("nan")
    except Exception as e:
        print(f"PyTorch failed for seq_len={seq_len}, d_head={d_head}, precision={precision}: {e}")
        pytorch_fwd_latency, pytorch_bwd_latency, pytorch_fwdbwd_latency = float("nan"), float("nan"), float("nan")

    # --- Triton Benchmark ---
    try:
        # Forward pass
        triton_fwd_latency = triton.testing.do_bench(lambda: triton_attention(q, k, v, causal), quantiles=quantiles)[0]

        # The backward pass for Triton is not implemented, so we'll only benchmark the forward pass.
        triton_bwd_latency, triton_fwdbwd_latency = float("nan"), float("nan")

    except torch.cuda.OutOfMemoryError as e:
        print(f"Triton OOM for seq_len={seq_len}, d_head={d_head}, precision={precision}: {e}")
        triton_fwd_latency, triton_bwd_latency, triton_fwdbwd_latency = float("nan"), float("nan"), float("nan")
    except Exception as e:
        print(f"Triton failed for seq_len={seq_len}, d_head={d_head}, precision={precision}: {e}")
        triton_fwd_latency, triton_bwd_latency, triton_fwdbwd_latency = float("nan"), float("nan"), float("nan")

    return {
        "seq_len": seq_len,
        "d_head": d_head,
        "precision": precision,
        "pytorch_fwd_ms": pytorch_fwd_latency,
        "pytorch_bwd_ms": pytorch_bwd_latency,
        "pytorch_fwdbwd_ms": pytorch_fwdbwd_latency,
        "triton_fwd_ms": triton_fwd_latency,
        "triton_bwd_ms": triton_bwd_latency,
        "triton_fwdbwd_ms": triton_fwdbwd_latency,
    }


def main():
    results = []

    # Parameter sweep
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_heads = [16, 32, 64, 128]
    precisions = ["bfloat16", "float32"]

    for seq_len in seq_lens:
        for d_head in d_heads:
            for precision in precisions:
                res = benchmark(seq_len, d_head, precision)
                results.append(res)
                print(f"Finished: seq_len={seq_len}, d_head={d_head}, precision={precision}")

    # Create and print DataFrame
    df = pd.DataFrame(results)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n--- Benchmark Results ---")
    print(df)

    # Save results to a CSV file
    df.to_csv("flash_attention_benchmark_results.csv", index=False)
    print("\nResults saved to flash_attention_benchmark_results.csv")


if __name__ == "__main__":
    main()
