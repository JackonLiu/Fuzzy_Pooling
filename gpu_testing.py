import torch

if torch.cuda.is_avialable():
 print("GPU is available")
else:
 print("Gpu NOT running")
