from collections.abc import Callable
import os
from typing import BinaryIO, Union
import torch
from torch import nn
from einops import einsum, reduce, rearrange
import math
from torch import Tensor
import numpy as np
import click
from jaxtyping import Float, Int, Bool


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        """
        Construct a linear transformation module
        Args:
            in_features (int): final dimension of input
            out_features (int): final dimension of output
            device (torch.device, optional): device to create the module on.
            dtype (torch.dtype, optional): data type of the module parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=True
        )

        # initialize the parameter to normal distribution with mean 0, sigma^2 = 2/(d_in + d_out) truncated to [-3sigma, 3sigma]
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct an embedding layer
        Args:
            num_embeddings (int): size of the vocabulary
            embedding_dim (int): dimension of embedding vectors
            device (torch.device, optional): device to create the module on.
            dtype (torch.dtype, optional): data type of the module parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # initialize the embedding matrix
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), requires_grad=True
        )
        # use normal distribution with mean 0, std = 1, truncated to [-3, 3]
        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer
        Args:
            indices (torch.Tensor): indices of the tokens to be embedded
        Returns:
            torch.Tensor: embedded vectors for the input indices
        """
        return self.weight[indices]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Construct a Root Mean Square Layer Normalization module
        Args:
            d_model (int): hidden dimension of the model
            eps (float): small value for numerical stability
            device (torch.device, optional): device to create the module on.
            dtype (torch.dtype, optional): data type of the module parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer
        Args:
            x (torch.Tensor): input tensor to be normalized
        Returns:
            torch.Tensor: normalized tensor
        """
        in_dtype = x.dtype

        # upscale to float32 for numerical stability and make sure squares dont overflow
        x = x.to(torch.float32)

        variance = reduce(x**2, "batch_size sequence_length d_model -> batch_size sequence_length", "mean")
        denominator = torch.rsqrt(variance + self.eps)
        denominator = rearrange(denominator, "batch_size sequence_length -> batch_size sequence_length 1")

        result = x * denominator * self.weight

        return result.to(in_dtype)  # convert back to original dtype


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a SwiGLU Feed-Forward Network module
        Args:
            d_model (int): input dimension
            d_ff (int): hidden dimension for the feed-forward network
            device (torch.device, optional): device to create the module on.
            dtype (torch.dtype, optional): data type of the module parameters
        """
        super().__init__()
        if not d_ff:
            d_ff = 8 / 3 * d_model  # hidden dimension is 8/3 times the input dimension
            # ensure it is multiple of 64
            d_ff = math.ceil(d_ff / 64) * 64
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

    def _swilu(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU activation function
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor after applying Swish
        """
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU layer
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor after applying SwiGLU
        """
        swish = self._swilu(self.w1(x))

        glu_part = swish * self.w3(x)
        result = self.w2(glu_part)
        return result


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Construct RoPE module
        Args:
            theta (float): the constant for the angle calculation
            d_k (int): dimension of the key vectors and query vectors
            max_seq_len (int): maximum sequence length for which the RoPE is computed
            device (torch.device, optional): device to create the module on.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        self.register_buffer("sin", torch.empty(max_seq_len, d_k // 2, device=device), persistent=False)
        self.register_buffer("cos", torch.empty(max_seq_len, d_k // 2, device=device), persistent=False)

        # fill the sine and cosine buffers
        self._fill_buffers()

    def _fill_buffers(self):
        """
        Fill the sine and cosine buffers with the appropriate values
        Args:
            max_seq_len (int): maximum sequence length for which the RoPE is computed
            theta (float): the constant for the angle calculation
        """
        positions = torch.arange(0, self.max_seq_len, device=self.device)
        # 2*k/ d_k
        denominator = torch.pow(
            self.theta,
            torch.arange(0, self.d_k // 2, device=self.device) * 2 / self.d_k,
        )
        # theta_i_k = positions[i] / denominator[k]
        theta = rearrange(positions, "seq_len -> seq_len 1") / rearrange(denominator, "d_k -> 1 d_k")

        # fill the sine and cosine buffers
        self.sin.copy_(torch.sin(theta))
        self.cos.copy_(torch.cos(theta))

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Forward pass of the RoPE layer
        Args:
            x (torch.Tensor): input tensor to be rotated
            token_positions (torch.Tensor): positions of the tokens in the sequence.
            We cannot always assume that the positions are sequential from 0, especially in inference.
        Returns:
            torch.Tensor: rotated tensor
        """
        sin_values: Float[Tensor, "... seq_len half_d_k"] = self.sin[token_positions]
        cos_values: Float[Tensor, "... seq_len half_d_k"] = self.cos[token_positions]

        x_pairs = rearrange(x, "... seq_len (half_d_k b) -> ... seq_len half_d_k b", b=2)

        x_0: Float[Tensor, "... seq_len half_d_k"] = x_pairs[..., 0]
        x_1: Float[Tensor, "... seq_len half_d_k"] = x_pairs[..., 1]

        x_0_rotated: Float[Tensor, "... seq_len half_d_k"] = x_0 * cos_values - x_1 * sin_values
        x_1_rotated: Float[Tensor, "... seq_len half_d_k"] = x_0 * sin_values + x_1 * cos_values

        x_rotated = rearrange([x_0_rotated, x_1_rotated], "p ... seq_len half_d_k -> ... seq_len (half_d_k p)")
        return x_rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute softmax along a specified dimension.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension of the input tensor along which to compute softmax.
    Returns:
        torch.Tensor: Softmax output tensor.
    """
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    keys: Float[Tensor, "batch_size ... seq_len d_k"],
    queries: Float[Tensor, "batch_size ... seq_len d_k"],
    values: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "batch_size ... queries d_v"]:
    pre_softmax = einsum(queries, keys, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        # mask is True for positions to keep, False for positions to mask, so we reverse
        mask = ~mask
        # add -inf to masked positions
        pre_softmax = pre_softmax.masked_fill(mask, float("-inf"))

    # Scale by sqrt(d_k)
    pre_softmax = pre_softmax / torch.sqrt(torch.tensor(keys.shape[-1], dtype=keys.dtype, device=keys.device))

    after_softmax = softmax(pre_softmax, dim=-1)
    return einsum(after_softmax, values, "... queries seq_len, ... seq_len d_v -> ... queries d_v")


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        token_positions: Int[Tensor, "... seq_len"] | None = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        """
        Construct a multi-head causal self-attention module
        Args:
            d_model (int): dimension of the model
            num_heads (int): number of attention heads
            theta (float, optional): constant for the RoPE, if None, RoPE is not used
            device (torch.device, optional): device to create the module on.
            dtype (torch.dtype, optional): data type of the module parameters
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.dtype = dtype
        self.token_positions = token_positions

        self.q_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=d_model, out_features=num_heads * self.d_k, device=device, dtype=dtype)

        self.output_proj = Linear(
            in_features=self.num_heads * self.d_k, out_features=self.d_model, device=device, dtype=dtype
        )

        self.rope = None

        if theta is not None:
            assert max_seq_len is not None, "max_seq_len must be provided if theta is specified"
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of the multi-head causal self-attention layer
        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_model)
        Returns:
            torch.Tensor: output tensor of shape (..., seq_len, d_model)
        """

        queries = rearrange(
            self.q_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads
        )

        keys = rearrange(
            self.k_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads
        )

        values = rearrange(
            self.v_proj(x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads
        )

        if self.rope:
            # apply rope to queries and keys
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=self.device, dtype=torch.int32)
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        # get mask with tril
        # mask is True for positions to keep, False for positions to mask
        mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2]), device=self.device, dtype=torch.bool), diagonal=0)

        # compute attention
        attention_output = scaled_dot_product_attention(
            keys=keys,
            queries=queries,
            values=values,
            mask=mask,  # no mask for causal self-attention
        )

        attention_output = rearrange(
            attention_output, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)", num_heads=self.num_heads
        )

        return self.output_proj(attention_output)


class PreNormTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a pre-norm transformer block
        Args:
            d_model (int): dimension of the model
            num_heads (int): number of attention heads
            d_ff (int): hidden dimension for the position wise feed-forward network
            theta (float, optional): rotary embedding parameter
            max_seq_len (int, optional): maximum sequence length
            device (torch.device, optional): device to create the module on.
            dtype (torch.dtype, optional): data type of the module parameters
        """
        super().__init__()
        self.attn = MultiHeadCausalSelfAttention(
            d_model=d_model, num_heads=num_heads, device=device, dtype=dtype, theta=theta, max_seq_len=max_seq_len
        )
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self, x: Float[Tensor, "... seq_len d_model"], token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of the pre-norm transformer block
        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len, d_model)
            token_positions (torch.Tensor): positions of the tokens in the sequence.
        Returns:
            torch.Tensor: output tensor of shape (..., seq_len, d_model)
        """
        rms_normed = self.ln1(x)
        attention_output = self.attn(rms_normed, token_positions=token_positions)
        residual_attention = x + attention_output

        rms_normed_ffn = self.ln2(residual_attention)
        ffn_output = self.ffn(rms_normed_ffn)
        final_output = residual_attention + ffn_output

        return final_output


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        num_layers: int,
        num_heads: int = 8,
        d_ff: int | None = None,
        rope_theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Construct a Transformer Language Model
        Args:
            vocab_size (int): size of the vocabulary
            d_model (int): dimension of the model
            context_length (int): maximum context length for the model to determine the dimensionality of the
                                  position embedding matrix
            num_layers (int): number of transformer blocks to use
            num_heads (int): number of attention heads in each transformer block
            rope_theta (float, optional): constant for the RoPE, if None, RoPE is not used
            device (torch.device, optional): device to create the module on.
            dtype (torch.dtype, optional): data type of the module parameters
        """
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        if d_ff is None:
            d_ff = (8 / 3) * d_model  # default hidden dimension for feed-forward network
            d_ff = math.ceil(d_ff / 64) * 64

        self.layers = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(
        self, x: Int[Tensor, "... seq_len"], token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        """
        Forward pass of the Transformer Language Model
        Args:
            x (torch.Tensor): input tensor of shape (..., seq_len) containing token indices
            token_positions (torch.Tensor): positions of the tokens in the sequence.
        Returns:
            torch.Tensor: output tensor of shape (..., seq_len, vocab_size) containing probabilities for
            each token in the vocabulary
        """

        embedded_tokens = self.token_embeddings(x)
        transformer_layer_output = embedded_tokens
        for layer in self.layers:
            transformer_layer_output = layer(transformer_layer_output, token_positions=token_positions)

        normed_output = self.ln_final(transformer_layer_output)
        lm_output = self.lm_head(normed_output)

        return lm_output


def cross_entropy_loss(
    predicted_logits: Float[Tensor, "b vocab_size"],
    target_indices: Int[Tensor, " b"],
) -> Float[Tensor, ""]:
    """
    Compute the avg cross-entropy loss for the predicted logits and target indices.
    l_i = -log(softmax(predicted_logits for the target_indices))
    Args:
        predicted_logits (torch.Tensor): Predicted logits of shape (batch_size, vocab_size).
        target_indices (torch.Tensor): Target indices of shape (batch_size,).
    Returns:
        torch.Tensor: Cross-entropy loss value (scalar).
    """
    # Use log-sum-exp trick for numerical stability
    # Extract target logits
    target_logits = predicted_logits[range(predicted_logits.shape[0]), target_indices]

    # Compute log-sum-exp: log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    max_logits = reduce(predicted_logits, "b vocab_size -> b 1", "max")
    exp_shifted = torch.exp(predicted_logits - max_logits)
    sum_exp = reduce(exp_shifted, "b vocab_size -> b", "sum")
    log_sum_exp = max_logits + torch.log(sum_exp)

    # Cross-entropy: -target_logit + log_sum_exp

    return (-target_logits + log_sum_exp).mean()


class AdamW(torch.optim.Optimizer):
    """
    AdamW optimizer.
    Formula:
    m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
    v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
    theta_t = theta_{t-1} - alpha * m_t / (sqrt(v_t) + eps) - alpha * lamda * theta_{t-1}
    where:
    - m_t is the first moment (mean of gradients)
    - v_t is the second moment (variance of gradients)
    - g_t is the gradient at time t
    - theta_t is the parameter at time t
    - alpha is the learning rate
    - eps is a small value for numerical stability
    - lamda is the weight decay factor
    Args:
        torch.optim.Optimizer: Base class for all optimizers in PyTorch.
    """

    def __init__(self, params, lr: float, betas: tuple[float], eps: float, weight_decay: float):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        """
        Perform a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]  # get the state of the parameter
                time_step = state.get("t", 1)  # get the iteration count for the parameter
                moment = state.get("m", torch.zeros_like(param.data))
                second_moment = state.get("v", torch.zeros_like(param.data))

                # update first and second moments
                gradient = param.grad.data
                moment = betas[0] * moment + (1 - betas[0]) * gradient
                second_moment = betas[1] * second_moment + (1 - betas[1]) * gradient**2

                alpha_t = lr * math.sqrt(1 - betas[1] ** time_step) / (1 - betas[0] ** time_step)

                # update the parameter
                param.data = param.data - alpha_t * moment / (second_moment.sqrt() + eps)
                # apply weight decay
                param.data = param.data - lr * weight_decay * param.data

                # update the states
                state["t"] = time_step + 1  # increment the iteration count for the parameter
                state["m"] = moment
                state["v"] = second_moment

        return loss  # return the loss if closure is provided, else None


def cosine_annealing_lr_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    warmup_steps: int,
    cosine_annealing_steps: int,
) -> float:
    """
    Compute the learning rate using cosine annealing schedule with warmup.
    Args:
        t (int): current step in the training process
        alpha_max (float): maximum learning rate
        alpha_min (float): minimum learning rate
        warmup_steps (int): number of steps for the warmup phase
        cosine_annealing_steps (int): number of steps for the cosine annealing phase
    Returns:
        float: learning rate for the current step
    """
    if t < warmup_steps:
        return alpha_max * t / warmup_steps
    elif warmup_steps <= t <= cosine_annealing_steps:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (
            1 + math.cos(((t - warmup_steps) / (cosine_annealing_steps - warmup_steps)) * math.pi)
        )
    return alpha_min  # after the cosine annealing phase, return the minimum learning rate


def gradient_clipping(
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> None:
    """
    Clip gradients of the parameters to prevent exploding gradients.
    If the combined l2 norm of all gradients exceeds max_norm, all gradients are scaled down
    by a factor of (max_norm / total_norm)
    Args:
        parameters (list[torch.nn.Parameter]): List of parameters to clip gradients for.
        max_norm (float): Maximum norm for the gradients.
    """
    # Calculate total norm across all parameters
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.data.norm(2) ** 2
    total_norm = total_norm**0.5

    # If total norm exceeds max_norm, scale all gradients
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    return None


def load_data(
    x: np.ndarray, batch_size: int, context_len: int, device: torch.device
) -> tuple[Int[Tensor, " batch_size context_len"], Int[Tensor, " batch_size context_len"]]:
    """
    Load data into batches for training.
    Args:
        x (np.ndarray): Input numpy array with tokens.
        batch_size (int): Size of each batch.
        context_len (int): Length of the context for each sample.
        device (torch.device): Device to load the data onto.
    Returns:
        tuple: A tuple containing:
            - Int[Tensor, " batch_size context_len"]: Input tensor of shape (batch_size, context_len).
            - Int[Tensor, " batch_size context_len"]: Target tensor of shape (batch_size, context_len).
    """

    # choose b random starting indices for the batches from 0 to len(x) - context_len
    start_indices = np.random.randint(0, len(x) - context_len, size=batch_size)

    # create input and target tensors
    input_tensor = torch.tensor(
        [x[start_idx : start_idx + context_len] for start_idx in start_indices],
        dtype=torch.int32,
        device=device,
    )
    # for each start index, the target is the next token in the sequence
    target_tensor = torch.tensor(
        [x[start_idx + 1 : start_idx + context_len + 1] for start_idx in start_indices],
        dtype=torch.int32,
        device=device,
    )
    return input_tensor, target_tensor


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: BinaryIO | str | os.PathLike[str]
) -> None:
    """
    Save the model and optimizer state to a checkpoint file.
    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): Current iteration number for naming the checkpoint file.
        out (Union[BinaryIO, str, os.PathLike[str]]): Output path or file-like object to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: BinaryIO | str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load the model and optimizer state from a checkpoint file.
    Args:
        src (Union[BinaryIO, str, os.PathLike[str]]): Source path or file-like object to load the checkpoint from.
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    Returns:
        int: The iteration number from the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]  # return the iteration number from the checkpoint


@click.command()
@click.option("--dataset", type=str, required=True, help="Path to the dataset file.")
@click.option("--batch_size", type=int, default=32, help="Batch size for training.")
@click.option("--vocab_size", type=int, default=10000, help="Size of the vocabulary.")
@click.option("--context_length", type=int, default=1024, help="Context length for the model.")
@click.option("--d_model", type=int, default=512, help="Dimension of the model")
@click.option("--num_layers", type=int, default=6, help="Number of transformer layers.")
@click.option("--num_heads", type=int, default=8, help="Number of attention heads.")
@click.option(
    "--d_ff",
    type=int,
    default=None,
    help="Hidden dimension for the feed-forward network. If None, defaults to 8/3 * d_model.",
)
@click.option("--rope_theta", type=float, default=None, help="Constant for the RoPE. If None, RoPE is disabled.")
@click.option("--device", type=str, default="cpu", help="Device to run the training on (e.g., 'cpu' or 'cuda').")
@click.option("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
@click.option("--iterations", type=int, default=1000, help="Number of training iterations.")
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
    iterations: int = 1000,
):
    """
    Train a Transformer Language Model on the given dataset.
    """
    device = torch.device(device)

    # Load dataset
    x = np.memmap(dataset, dtype=np.int32, mode="r")
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

    for iteration in range(iterations):
        # Load data
        input_tensor, target_tensor = load_data(x, batch_size, context_length, device)

        # Forward pass
        logits = model(input_tensor)

        # Compute loss
        loss = cross_entropy_loss(logits.view(-1, vocab_size), target_tensor.view(-1))

        lr = cosine_annealing_lr_schedule(
            t=iteration,
            alpha_max=1e-1,
            alpha_min=1e-5,
            warmup_steps=100,
            cosine_annealing_steps=iterations,
        )
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_norm=4.0)
        optimizer.step()

        print(f"Iteration {iteration + 1}/{iterations}, Loss: {loss.item()}")

        # Save checkpoint every 100 iterations
        if (iteration + 1) % 100 == 0:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f"checkpoint_{iteration + 1}.pt")
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1e3)

    for _ in range(100):
        opt.zero_grad()  # reset gradients for all learning parameters
        loss = (weights**2).mean()  # compute the loss
        print(f"Loss: {loss.item()}")
        loss.backward()  # compute gradients
        opt.step()  # update parameters using the optimizer
