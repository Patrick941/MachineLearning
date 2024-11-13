import torch

# Check if GPU is accessible
print("Is ROCm GPU available:", torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
