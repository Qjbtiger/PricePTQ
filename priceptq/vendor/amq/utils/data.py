# Adapted for the PricePTQ paper release.
# Import necessary modules
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def get_tokenizer(model, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, cache_dir=cache_dir)
    return tokenizer



def get_wikitext2_trainenc(seed, n_sample, tokenizer, batch_size=1, seqlen=2048, cache_dir=None):
    
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
    traindata = traindata.shuffle(seed=seed)
    trainenc = tokenizer("\n\n".join(traindata[:n_sample]['text']), return_tensors='pt').input_ids
    n_sample = trainenc.numel() // seqlen
    trainenc = trainenc[:, :n_sample * seqlen].reshape(n_sample, seqlen)

    return DataLoader(trainenc, batch_size=batch_size)



def get_loader(name, n_sample=128, train=True, seed=0, seqlen=2048, tokenizer=None, model='', cache_dir=None):
    """Build the WikiText2 calibration loader used during AMQ search."""
    if name != 'wikitext2' or not train:
        raise ValueError("Unsupported loader configuration; expected name='wikitext2' with train=True.")
    tokenizer = tokenizer or get_tokenizer(model, cache_dir=cache_dir)
    return get_wikitext2_trainenc(seed=seed, n_sample=n_sample, seqlen=seqlen,
                                 tokenizer=tokenizer, cache_dir=cache_dir)
