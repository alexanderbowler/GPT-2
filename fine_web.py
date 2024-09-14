""" 
Loads the FineWeb edu dataset from hugging face
Saves shards of the dataset to local dir "edu_fineweb10B"
"""
import os
import multiprocessing as mp
import numpy as np 
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm

#-------------------------------------------------#-------------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard total 100 shards

# create cache for local dir if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download dataset from internet
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init tokenizer
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens["<|endoftext|>"] # end of text token
def tokenize(doc):
    # tokenizes a document and returns the tokens as an array of np.uint16
    tokens = [eot] # this token begins a document
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # writes np array of tokens as binary file
    np.save(filename, tokens_np)

# tokenize all documents and save all output shards, each of shard_size number of tokens
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens
        if token_count + len(tokens) < shard_size:
            # append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
            progress_bar.update(len(tokens))
        else:
        # write the current shard and start a new one
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(DATA_CACHE_DIR, f"edu_fineweb_{split}_{shard_index:06d}")
            # split documents into whatever fits into this shard
            remainder = shard_size - token_count
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            #populate next shard with leftovers from current doc
            token_count = len(tokens) - remainder
            all_tokens_np[:token_count] = tokens[remainder:]
    
    # write any remaining tokens as last shard
    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = os.path.join(DATA_CACHE_DIR, f"edu_fineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count]) 
