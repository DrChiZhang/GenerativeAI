import os
from os.path import exists
import torch
from models import *  # Assumes `make_model`, `subsequent_mask`, etc. are defined here

# ----------------------------
# Greedy decoding function
# ----------------------------
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    # Run the encoder to get memory representation of the source
    memory = model.encode(src, src_mask)
    
    # Initialize the output sequence with the start symbol
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    # Generate tokens one by one up to max_len
    for i in range(max_len - 1):
        # Decode using current ys and memory
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # Get probabilities for the last generated token
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)  # Select token with highest probability
        next_word = next_word.data[0]

        # Append predicted token to the sequence
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

# ----------------------------
# Launch distributed training across GPUs
# ----------------------------
def train_distributed_model(tokenizer_src, tokenizer_tgt, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"  # Required by PyTorch's DDP
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    
    # Spawn a training process on each GPU
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, tokenizer_src, tokenizer_tgt, config, True),
    )

# ----------------------------
# Train on a single GPU or use DDP if specified
# ----------------------------
def train_model(tokenizer_src, tokenizer_tgt, config):
    if config["distributed"]:
        train_distributed_model(tokenizer_src, tokenizer_tgt, config)
    else:
        train_worker(0, 1, tokenizer_src, tokenizer_tgt, config, False)

# ----------------------------
# Load trained model or trigger training if not found
# ----------------------------
def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }

    model_path = "./ckpt/multi30k_model_final.pt"

    print("🔠 Loading tokenizers...")
    tokenizer_src, tokenizer_tgt = load_tokenizers()
    
    # Train model if not already saved
    if not exists(model_path):
        train_model(tokenizer_src, tokenizer_tgt, config)

    # Create and load the trained model weights
    model = make_model(len(tokenizer_src), len(tokenizer_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model

def main():
    # Load the model and tokenizer
    model = load_trained_model()
    
    # Example input for inference
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    
    # Generate predictions using greedy decoding
    ys = greedy_decode(model, src, src_mask, max_len=20, start_symbol=1)
    
    print("Example Untrained Model Prediction:", ys)

if __name__ == "__main__":
    main()