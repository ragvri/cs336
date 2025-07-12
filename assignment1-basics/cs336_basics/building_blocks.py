import torch
from torch import nn
from einops import einsum, reduce, rearrange
import math
from torch import Tensor
from jaxtyping import Float, Int


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
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=True)

        # initialize the parameter to normal distribution with mean 0, sigma^2 = 2/(d_in + d_out) truncated to [-3sigma, 3sigma]
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")


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
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), requires_grad=True
        )
        # use normal distribution with mean 0, std = 1, truncated to [-3, 3]
        std = 1.0
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the embedding layer
        Args:
            indices (torch.Tensor): indices of the tokens to be embedded
        Returns:
            torch.Tensor: embedded vectors for the input indices
        """
        return self.embedding_matrix[indices]


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
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)

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

        result = x * denominator * self.gain

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
        self.linear1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

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
        swish = self._swilu(self.linear1(x))

        glu_part = swish * self.linear3(x)
        result = self.linear2(glu_part)
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


if __name__ == "__main__":
    # Example usage
    linear_layer = Linear(in_features=4, out_features=2)
    # get state dict for weights
    print(linear_layer.state_dict())
