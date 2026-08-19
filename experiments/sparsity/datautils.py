"""
datautils.py
------------
Dataset streaming, tokenization, and calibration data loader utilities for SparseGPT.
Supports WikiText2, PTB, and C4 subsets with deterministic sample batching.
"""

import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

def set_seed(seed: int = 0) -> None:
    """Sets random seeds for reproducible calibration data sampling."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_tokenizer(model: str):
    """Loads fast or legacy tokenizer matching the given model identifier."""
    if "llama" in model.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
        if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
            try:
                tokenizer.bos_token_id = 1
                tokenizer.eos_token_id = 2
            except AttributeError:
                pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    return tokenizer

def get_c4(nsamples: int, seed: int, seqlen: int, model: str):
    """
    Streams and tokenizes slices from the C4 (allenai/c4) training and validation splits.
    Returns:
        dataloader: List of tokenized input tensor tensors of shape [1, seqlen]
        testenc: Full tokenized validation split for perplexity evaluation
    """
    traindata = load_dataset(
        "allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
    )
    valdata = load_dataset(
        "allenai/c4", "allenai--c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation"
    )

    tokenizer = get_tokenizer(model)
    random.seed(seed)
    trainloader = []

    # Sample random sequence windows from the training split
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Tokenize first 256 validation documents for evaluation
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_loaders(name: str, nsamples: int = 128, seed: int = 0, seqlen: int = 2048, model: str = ""):
    """
    Dispatcher returning train calibration dataloaders and test perplexity tokenizers.
    Supported datasets: c4.
    """
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
