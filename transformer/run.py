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
def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"  # Required by PyTorch's DDP
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    
    # Spawn a training process on each GPU
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )

# ----------------------------
# Train on a single GPU or use DDP if specified
# ----------------------------
def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)

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

    model_path = "multi30k_model_final.pt"
    
    # Train model if not already saved
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    # Create and load the trained model weights
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model

# ----------------------------
# Automatically load model in Jupyter notebooks
# ----------------------------
if is_interactive_notebook():
    model = load_trained_model()

# ----------------------------
# Visualize and compare model outputs with target
# ----------------------------
def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)

        # Load one example
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)

        # Decode prediction
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        # Convert input and target token ids to strings
        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))

        # Decode and convert model output to tokens
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join([vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx])
            .split(eos_string, 1)[0]
            + eos_string
        )

        print("Model Output               : " + model_txt.replace("\n", ""))

        # Store results
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results

# ----------------------------
# Main function: entry point for testing
# ----------------------------
def main(n_examples=5):
    global vocab_src, vocab_tgt

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        batch_size=1
    )

    print("Loading Trained Model ...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data

# ----------------------------
# Run script if executed directly
# ----------------------------
if __name__ == "__main__":
    main()
