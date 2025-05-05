import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad

from .coder import *  # Import model architecture builder (e.g., Transformer)
from .dataset import create_dataloaders  # Function to load training and validation dataloaders

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank> token index
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # Mask to ignore padding in source
        if tgt is not None:
            self.tgt = tgt[:, :-1]  # Decoder input (shifted right)
            self.tgt_y = tgt[:, 1:]  # Target for loss computation (shifted left)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)  # Combine padding & future masking
            self.ntokens = (self.tgt_y != pad).data.sum()  # Count of non-padding tokens

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)  # Padding mask
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)  # Future mask
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # Total # of examples used
    tokens: int = 0  # Total # of tokens processed

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Run a single epoch for training or evaluation."""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)  # Forward pass
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)  # Compute loss

        if mode == "train" or mode == "train+log":
            loss_node.backward()  # Backward pass
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                optimizer.step()  # Optimizer step
                optimizer.zero_grad(set_to_none=True)  # Reset gradients
                n_accum += 1
                train_state.accum_step += 1

            scheduler.step()  # Learning rate scheduler step

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        # Logging every 40 steps
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0

        del loss
        del loss_node

    return total_loss / total_tokens, train_state

class LabelSmoothing(nn.Module):
    """Implement label smoothing to soften target distributions."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # Distribute smoothing
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)  # Set true target prob
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())  # KL divergence

class SimpleLossCompute:
    """A wrapper that computes loss and scales it by number of tokens."""

    def __init__(self, generator, criterion):
        self.generator = generator  # Final linear + softmax layer
        self.criterion = criterion  # LabelSmoothing or CrossEntropy

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

def train_worker(
    gpu,
    ngpus_per_node,
    tokenizer_src,
    tokenizer_tgt,
    config,
    is_distributed=False,
):
    """Function for training on a specific GPU (supports distributed)."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Train worker process using Device: {device} for training", flush=True)

    pad_idx = tokenizer_tgt.pad_token_id
    d_model = 512  # Model dimensionality
    model = make_model(len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab()), N=6)
    model.to(device)
    module = model
    is_main_process = True

    if is_distributed:
        dist.init_process_group("nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node)
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(tokenizer_tgt.get_vocab()), padding_idx=pad_idx, smoothing=0.1).to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        tokenizer_src,
        tokenizer_tgt,
        batch_size=config["batch_size"] // ngpus_per_node, 
        device=device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b["src"], b["tgt"], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b["src"], b["tgt"], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)
