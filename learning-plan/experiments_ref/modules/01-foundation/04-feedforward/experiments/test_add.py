import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeedForward(nn.Module):
    def __init__(self, dim=512, intermediate=2048):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        print(f"gate: {gate.shape}")

        up = self.up_proj(x)
        print(f"up: {up.shape}")

        hidden = F.silu(gate) * up
        print(f"hidden: {hidden.shape}")

        output = self.down_proj(hidden)
        print(f"output: {output.shape}")

        return output

# 测试
ffn = SimpleFeedForward()
x = torch.randn(2, 10, 512)  # [batch=2, seq=10, dim=512]
print(f"input: {x.shape}")
output = ffn(x)

x = torch.randn(1, 5, 512)  # 5 个 token
gate = F.silu(ffn.gate_proj(x))  # [1, 5, 2048]

# 查看不同 token 的门控激活
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.hist(gate[0, i].detach().numpy(), bins=50)
    plt.title(f"Token {i}")
plt.suptitle("Gate Activations")
plt.show()