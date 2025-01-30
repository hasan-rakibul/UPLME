import torch

def is_gpu():
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 * 1014 * 1024)} GB")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()}")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()}")

if __name__ == "__main__":
    is_gpu()
