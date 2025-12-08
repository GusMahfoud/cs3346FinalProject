import torch
ckpt = torch.load("models/a2c_v24/checkpoint.pth", map_location="cpu")
print("Saved epsilon:", ckpt["epsilon"])