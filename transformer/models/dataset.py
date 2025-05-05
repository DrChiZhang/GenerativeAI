import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# Load Hugging Face tokenizers for source and target languages
def load_tokenizers(src_model_name="Helsinki-NLP/opus-mt-de-en", tgt_model_name=None):
    """
    Loads pre-trained tokenizers. By default, the same tokenizer is used for both source and target,
    which works for models like MarianMT.

    Args:
        src_model_name (str): Pre-trained model name for the source language.
        tgt_model_name (str): Optional. Pre-trained model name for the target language.

    Returns:
        tokenizer_src, tokenizer_tgt (AutoTokenizer): Tokenizer instances.
    """
    tokenizer_src = AutoTokenizer.from_pretrained(src_model_name)
    tokenizer_tgt = tokenizer_src if tgt_model_name is None else AutoTokenizer.from_pretrained(tgt_model_name)
    print(tokenizer_src.pad_token)      # Might print: <pad>
    print(tokenizer_tgt.pad_token_id)   # Might print: 2
    return tokenizer_src, tokenizer_tgt


# Tokenize a batch of source and target sentences
def tokenize_batch(batch, tokenizer_src, tokenizer_tgt, src_lang="de", tgt_lang="en"):
    """
    Tokenizes a batch of examples from the dataset using the specified tokenizers.

    Args:
        batch (dict): A batch of examples with keys 'de' and 'en'.
        tokenizer_src: Tokenizer for the source language.
        tokenizer_tgt: Tokenizer for the target language.

    Returns:
        dict: Tokenized input_ids, attention_mask, and labels.
    """
    src = tokenizer_src(batch[src_lang], truncation=True, padding=False)
    tgt = tokenizer_tgt(batch[tgt_lang], truncation=True, padding=False)
    return {
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "labels": tgt["input_ids"],
    }


# Pad all sequences in a batch to the same length
def collate_batch(batch, pad_token_id, device="cuda"):
    """
    Pads a batch of tokenized sequences so they can be fed into a model.

    Args:
        batch (list of dict): List of tokenized examples.
        pad_token_id (int): Token ID used for padding.

    Returns:
        dict: Batched and padded input_ids, attention_mask, and labels.
    """
    input_ids = [torch.tensor(example["input_ids"]).to(device) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"]).to(device) for example in batch]
    labels = [torch.tensor(example["labels"]).to(device) for example in batch]

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)  # 0 for attention mask
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    return {
        "src": input_ids,
        "tgt": labels,
        "attention_mask": attention_mask,
    }


# Load and tokenize the dataset, then return PyTorch DataLoaders
def create_dataloaders(tokenizer_src, tokenizer_tgt, batch_size=32, src_lang="de", tgt_lang="en", device="cuda"):
    """
    Loads the Multi30k dataset, tokenizes it using provided tokenizers, and returns DataLoaders.

    Args:
        tokenizer_src: Tokenizer for source language.
        tokenizer_tgt: Tokenizer for target language.
        batch_size (int): Batch size for DataLoader.
        src_lang (str): Source language key in dataset (default "de").
        tgt_lang (str): Target language key in dataset (default "en").

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation DataLoaders.
    """
    # Load the Multi30k dataset splits
    dataset = load_dataset("bentrevett/multi30k")

    tokenized_datasets = {}
    for split in ["train", "validation", "test"]:
        # Apply tokenization to the entire dataset
        tokenized = dataset[split].map(
            lambda x: tokenize_batch(x, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang),
            batched=True,
            remove_columns=dataset[split].column_names,  # Removes original text fields
        )
        tokenized.set_format(type="torch")  # Ensure tensors are returned
        tokenized_datasets[split] = tokenized

    # Define a collate function for padding batches
    pad_id = tokenizer_src.pad_token_id
    collate_fn = lambda batch: collate_batch(batch, pad_token_id=pad_id, device=device)

    # Create PyTorch DataLoaders
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader

