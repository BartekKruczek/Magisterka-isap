import torch

def list_gpus():
    num_gpus = torch.cuda.device_count()
    print(f"Liczba dostępnych GPU: {num_gpus}\n")
    for i in range(num_gpus):
        device = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {device.name}")
        print(f"  - Pojemność pamięci: {device.total_memory / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    list_gpus()
