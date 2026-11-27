# Adapted for the PricePTQ paper release.
import random
import os
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset





def get_gptq_calib_dataset(data="wikitext2", tokenizer=None, n_samples=128, seed=0, seqlen=2048):
    """Sample random WikiText2 training windows for GPTQ calibration."""
    if data != "wikitext2":
        raise ValueError("The paper GPTQ protocol uses WikiText2.")
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(n_samples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        inp = trainenc.input_ids[:, i:i + seqlen]
        target = inp.clone()
        target[:, :-1] = -100
        trainloader.append((inp, target))
    return trainloader


class BASE:
    def __init__(self, model, tokenizer, method, arch, avg_bits, group_size=128, config=None, dev='cuda', **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.arch = arch
        self.avg_bits = avg_bits
        self.group_size = group_size
        self.config = config
        self.dev = dev
        
        print(f'Quantization options: method: {self.method}, arch: {self.arch}, avg_bits: {self.avg_bits:.4f}, group_size: {self.group_size}, config: {self.config}, dev: {self.dev}')

    def append_str_prefix(self, x, prefix):
        if isinstance(x, str):
            return prefix + x
        elif isinstance(x, tuple):
            return tuple([self.append_str_prefix(y, prefix) for y in x])
        elif isinstance(x, list):
            return [self.append_str_prefix(y, prefix) for y in x]
        else:
            return x

    @staticmethod
    def get_named_linears(module):
        return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}
    

    @staticmethod    
    def get_op_by_name(module, op_name):
        # get the op by its name relative to the module
        for name, m in module.named_modules():
            if name == op_name:
                return m
        raise ValueError(f"Cannot find op {op_name} in module {module}")


    @staticmethod
    def set_op_by_name(layer, name, new_module):
        levels = name.split(".")
        if len(levels) > 1:
            mod_ = layer
            for l_idx in range(len(levels) - 1):
                if levels[l_idx].isdigit():
                    mod_ = mod_[int(levels[l_idx])]
                else:
                    mod_ = getattr(mod_, levels[l_idx])
            setattr(mod_, levels[-1], new_module)
        else:
            setattr(layer, name, new_module)


    @staticmethod
    def get_op_name(module, op):
        # get the name of the op relative to the module
        for name, m in module.named_modules():
            if m is op:
                return name
        raise ValueError(f"Cannot find op {op} in module {module}")
