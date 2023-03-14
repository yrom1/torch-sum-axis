import torch

t = torch.arange(30).reshape(2, 3, 5)
print(f"{t=}")
print(f"{t.sum(0)=}")
print(f"{t.sum(1)=}")
print(f"{t.sum(0)=}")
