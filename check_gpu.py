import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Test actual GPU computation
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print(f"GPU tensor test passed. Result shape: {z.shape}")