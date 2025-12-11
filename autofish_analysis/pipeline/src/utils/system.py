import torch
import os


class ActSys:
    def check_gpu_availability(self):
        has_gpu = torch.cuda.is_available()
        gpu_device_name = ""
        if has_gpu:
            gpu_device_name = torch.cuda.get_device_name(0)  # Get name of the first GPU
            print(f"GPU detected: Yes ({gpu_device_name})")
            return True
        else:
            print("GPU detected: No. Running on CPU.")
            return False
