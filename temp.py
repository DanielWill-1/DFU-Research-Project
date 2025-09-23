# Run this locally once to download the model file
import torch
torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
print("Model downloaded. Check your cache folder.")
# On Windows, find it in: C:/Users/<YourUser>/.cache/torch/hub/intel-isl_MiDaS_master
# On Linux/Mac, find it in: ~/.cache/torch/hub/intel-isl_MiDaS_master