from torch import nn
import torch


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


data = torch.randn(32, 100, dtype=torch.float32)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    model = ToyModel(100, 10).cuda()
    # print dtype of model parameters
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, dtype: {param.dtype}")

    # get dtype of model.fc1 output
    output = model.fc1(data.cuda())
    print(f"fc1 dtype: {output.dtype}")

    # get dtype of model.ln output
    output = model.ln(output)
    print(f"ln dtype: {output.dtype}")

    # logits dtype
    logits = model(data.cuda())
    print(f"Logits dtype: {logits.dtype}")

    # grad dtype
    logits.mean().backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradients for {name}, dtype: {param.grad.dtype}")
