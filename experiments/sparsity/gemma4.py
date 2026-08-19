"""
gemma4.py
---------
Structured and unstructured SparseGPT pruning implementation for Google Gemma 4 models.

Attribution Notice:
    Adapted from SparseGPT (Frantar & Alistarh, 2023 - https://github.com/IST-DASLab/sparsegpt).
    Modifications for Gemma 4 architecture and Cloud TPU acceleration include:
      - Support for Gemma 4 hybrid attention (full attention and sliding window causal masks).
      - Rotary embedding caching across distinct unique layer types.
      - Support for shared KV / MLA projection weights on every 6th layer.
      - Offloading layer-wise Hessian computation and N:M structured projection to TPU MXUs.
"""

import argparse
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration, AutoConfig

from .tpu_sparsegpt import TPUSparseGPT as SparseGPT
from .modelutils import find_layers
from .quant import Quantizer
from .datautils import get_loaders

def get_gemma4(model_path: str) -> Gemma4ForConditionalGeneration:
    """Loads a Gemma 4 model with uninitialized random weight hooks disabled for fast instantiation."""
    print(f"Loading Gemma 4 from {model_path}...")
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.seqlen = 2048
    return model

@torch.no_grad()
def gemma4_sequential(model, dataloader, dev, args):
    """
    Performs layer-by-layer sequential SparseGPT pruning on Gemma 4.
    Passes calibration activations through each layer, captures empirical Hessians on TPU MXUs,
    and applies second-order error compensation.
    """
    print("Starting Gemma 4 sequential pruning...")

    lm = model.model.language_model
    layers = lm.layers
    embed_tokens = lm.embed_tokens.to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.text_config.hidden_size
    seqlen = model.seqlen

    # Pre-collect calibration embeddings from dataloader
    inps = torch.zeros((args.nsamples, seqlen, hidden_size), dtype=dtype, device=dev)
    position_ids = torch.arange(seqlen, device=dev).unsqueeze(0)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[self.cur_sample] = inp
            self.cur_sample += 1
            raise ValueError

    catcher = Catcher(layers[0])
    catcher.cur_sample = 0
    layers[0] = catcher

    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = catcher.module
    layers[0] = layers[0].cpu()
    embed_tokens = embed_tokens.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    # Precompute RoPE position embeddings for each unique attention layer type
    position_embeddings = {}
    for layer_type in lm.unique_layer_types:
        position_embeddings[layer_type] = lm.rotary_emb(inps[0:1], position_ids, layer_type)

    from transformers.models.gemma4.modeling_gemma4 import create_causal_mask, create_sliding_window_causal_mask
    causal_mask_mapping = {
        "full_attention": create_causal_mask(
            config=lm.config,
            inputs_embeds=inps[0:1],
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        ),
        "sliding_attention": create_sliding_window_causal_mask(
            config=lm.config,
            inputs_embeds=inps[0:1],
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        ),
    }
    for k in causal_mask_mapping:
        if causal_mask_mapping[k] is not None and torch.is_tensor(causal_mask_mapping[k]):
            causal_mask_mapping[k] = causal_mask_mapping[k].to(dev)

    # Process each Transformer decoder layer sequentially
    for i in range(len(layers)):
        layer_type = lm.config.layer_types[i]
        layer = layers[i].to(dev)

        pos_emb = position_embeddings[layer_type]
        attn_mask = causal_mask_mapping[layer_type]

        subset = find_layers(layer)
        gpts = {}

        if args.minlayer <= i < args.maxlayer:
            for name in subset:
                if args.prune_only and args.prune_only not in name:
                    continue
                gpts[name] = SparseGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in gpts:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    position_embeddings=pos_emb,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    shared_kv_states={},
                )[0]

            for h in handles:
                h.remove()

            for name in subset:
                if name in gpts:
                    print(f"Pruning {name} in layer {i} (2:4 OBS update)...")
                    gpts[name].fasterprune(
                        args.sparsity,
                        prunen=args.prunen,
                        prunem=args.prunem,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )
                    gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=position_ids,
                shared_kv_states={},
            )[0]

        layers[i] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    print("\nSparsification complete!")
