import torch
from torch import nn
from einops import einsum


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


if __name__ == "__main__":
    # Example usage
    linear_layer = Linear(in_features=4, out_features=2)
    # get state dict for weights
    print(linear_layer.state_dict())
    