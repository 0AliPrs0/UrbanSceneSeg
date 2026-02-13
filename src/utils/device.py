import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

        # Enable TF32 if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Speed optimization
        torch.backends.cudnn.benchmark = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS backend")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
