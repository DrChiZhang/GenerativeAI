import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    compute_capability = torch.cuda.get_device_capability(current_device)
    
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_capability}")

    # Check fp16 (half-precision) support
    fp16_supported = compute_capability >= (6, 0)  # Pascal or newer
    print(f"FP16 Supported: {fp16_supported}")

    # Check bf16 (bfloat16) support
    bf16_supported = compute_capability >= (8, 0)  # Ampere or newer
    print(f"BF16 Supported: {bf16_supported}")

    # Check if PyTorch was built with BF16 support
    try:
        bf16_available = torch.cuda.is_bf16_supported()
        print(f"PyTorch BF16 Support: {bf16_available}")
    except AttributeError:
        print("PyTorch version <1.11: BF16 check not available")
else:
    print("CUDA not available.")

if fp16_supported:
    try:
        a = torch.randn(2, 2, dtype=torch.float16).cuda()
        b = torch.randn(2, 2, dtype=torch.float16).cuda()
        c = torch.matmul(a, b)
        print("FP16 test succeeded.")
    except RuntimeError:
        print("FP16 test failed.")

if bf16_supported:
    try:
        a = torch.randn(2, 2, dtype=torch.bfloat16).cuda()
        b = torch.randn(2, 2, dtype=torch.bfloat16).cuda()
        c = torch.matmul(a, b)
        print("BF16 test succeeded.")
    except RuntimeError:
        print("BF16 test failed.")