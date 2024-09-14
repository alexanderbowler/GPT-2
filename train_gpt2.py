from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_layer == 0
        # key, query, and value projections together in a batch, is 3* bc is query, key, and value all of size n_embd
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # Note: everything named to match HF
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #flag for scaling
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias more of a mask to prevent the predicted words from looking at future words
        # that should not be seen, but called a bias by OPENAI/HF
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # Batch size, Token sequence length, and embedding dimensionality (channels) from input

        qkv = self.c_attn(x) 
        q, k, v = qkv.split(self.n_embd, dim = 2) #splits the concatenated k,q,v into each their own
        # rearrange dimensions so that the number of heads nh is a batch dimension to following operations
        # are applied in parallel by pytorch, hs is head size, and C (number of channels) = hs*nh
        # eg. in GPT2 hs = 64 nh = 12 so C = 64*12 = 768
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)


        # Attention the following does batchwise operations along the B dim and nh dim, to get (T,T) of full context
        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1))) # (B, nh, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # Masks upper right of matrix to 0s (after softmax) to prevent using words not yet seen
        # att = F.softmax(att, dim=-1)
        # y = att @ v #(B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs) back to original dim for concatentation

        #replace the above with flash attention with
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C) # reordered back to 
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #flag for scaling


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPT2Config:
    block_size: int = 1024 # max sequence token length
    vocab_size: int = 50257 # number of tokens 50,000 BPE merges + 256 bytes tokens + 1 <end of text> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights) # applies the initialize weights function to all modules within model (nn method)

    # copying gpt2 initialization schema
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 * std
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx (B, T) input
        B, T = idx.size()
        #print(idx)
        assert T <= self.config.block_size, f"Context length too long, cannot forward sequence of length {T}, blocksize is {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_embd = self.transformer.wpe(pos) # position embedding shape (T, n_embd)
        tok_embd = self.transformer.wte(idx) # token embedding shape (B, T, n_embd)
        x = tok_embd + pos_embd # adds position embedding to each batch
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward to final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss



    @classmethod
    def from_pretrained(cls, model_type): # acts as a constructor given the model type
        """Loads pretrained GPT2 model weights from HF"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large' 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        #n_layer, n_head, and n_embed are determined by model_type
        config_args = {
            'gpt2' :        dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium' : dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large' :  dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl' :     dict(n_layer=48, n_head=25, n_embd=1600) # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 vocab size for gpt2
        config_args['block_size'] = 1024 # always 1024 context length for gpt2
        # create an initialized from scratch gpt model
        config = GPT2Config(**config_args)
        #print(config)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys  = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the mask used for training

        # init a hugging face transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy over params making sure all the names align
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # ignore the bias mask
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore the bias mask
        # need to transpose some of the weights bc of their tensorflow format, bc openai uses conv1d module we want to use a vanilla module
        transposed = ['attn.c_attn.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight', 'attn.c_proj.weight']#[, , ', ]

        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for conv1d that we need to tranpose
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"shape mismatch {sd_hf[k].shape} != {sd[k].shape}, for key {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #vanilla copy
                assert sd_hf[k].shape == sd[k].shape, f"shape mismatch {sd_hf[k].shape} != {sd[k].shape}, for key {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate params
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        #create optim groups all params which are 2d will be decayed
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, 'weight_decay': weight_decay},
            {"params": nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decay parameter tensors: {len(decay_params)} with {num_decay_params:,} params")
            print(f"num no decay parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} params")
        # create the optimizer and fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused adamW: {used_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=used_fused)
        return optimizer
    
#-------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    '''
        Loads the data from tinyshakespeare into batches for the gpt 
    '''
    def __init__(self, T:int, B:int, process_rank:int, num_processes:int, split:str):
        self.T = T
        self.B = B
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert(split in {'train', 'val'})

        # get the shard filename
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(self.shards) > 0, f"no shards found in split {split}"
        if master_process:
            print(f"Found {len(self.shards)} shards in split {split}")
        self.reset()

    def reset(self):
        # init state
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # so each starts at unique place

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position+B*T+1].clone()
        x = buf[:-1].view(B,T) #inputs
        y = buf[1:].view(B,T) #targets
        # advance to next tokens
        self.current_position += B * T * self.num_processes
        # if batch would be at the end reset batch
        if self.current_position+(B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x,y
    
#-----------------------------------HellaSwag-------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

#-----------------------------------Main-----------------
# simple launch 
# python train_gpt2.py
# ddp launch
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
import time
import os

# run the training loop 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from hellaswag import iterate_examples, render_example

def main():

    # set up ddp (distributed data parallel)
    # the torchrun command will setup the enviroment vars RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # true if is ddp run

    if ddp:
        # use of ddp atm demands CUDA, we set device appropriately according to RANK
        assert torch.cuda.is_available(), "can only run ddp on CUDA"
        init_process_group('nccl')
        ddp_rank = int(os.environ['RANK']) # is unique for each subprocess which is running
        ddp_local_rank = int(os.environ['LOCAL_RANK']) # local rank is gpu number for current computer, can be same over diff computers
        ddp_world_size = int(os.environ['WORLD_SIZE']) # total number of gpus in system
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc
        if ddp_world_size == 1:
            device = 'cuda'

    else:
        # vanilla, single gpu training
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 0
        master_process = True
        # autodetect device
        device = (
                'cuda' if torch.cuda.is_available()
                else 'mps' if torch.backends.mps.is_available()
                else 'cpu'
            )
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding('gpt2')

    total_batch_size = 524288 # 2^19 ~.5M so good number
    B = 64 # micro batch size
    T = 1024 # context length
    assert total_batch_size % (T * B) == 0, "make sure total_batch_size is diviisble by T * B * ddp_world_size"
    if(ddp_world_size == 0):
        grad_accum_steps = total_batch_size // (T * B)
    else:
        grad_accum_steps = total_batch_size // (T * B * ddp_world_size)


    if master_process:
        print(f"Total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B = B, T = T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B = B, T = T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')


    torch.set_float32_matmul_precision('high') # this enables matmuls with tf32 instead of fp32, which chops the mantissa allowing faster compute and less mem

    # create model
    model = GPT(GPT2Config(vocab_size=50304)) # identical models created for each gpu bc of same seed
    model.to(device)
    use_compile = False
    if use_compile:
        model = torch.compile(model) # all models are compiled 
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always has the raw unwrapped model

    max_steps = 19073
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    def get_lr(it):
        # 1) linear warmup for learning rate
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay iters return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between use cosine decay
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1.0
        coef = 0.5 * (1 + math.cos(math.pi * decay_ratio)) #coeff starts at 1 goes to 0
        return min_lr + coef * (max_lr - min_lr)

    # optimize
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

    # logs
    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, 'w') as f: # open for writing to clear the file
        pass

    # training loop
    for step in range(max_steps):
        t0 = time.time() # for timing
        last_step = (step == max_steps - 1)

        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # allows activations to be in bfloat16, but params are left in float32
                        logits, loss = model(x, y)
                    loss = loss/val_loss_steps # to normalize the loss as it would be in 1 batch
                    val_loss_accum += loss.detach() 
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 1000 == 0 or last_step):
                    # optionally save model
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config' : raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, checkpoint_path)


        # once in a while evaluate hellaswag
        if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
            num_correct_norm = 0
            num_correct = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render examples into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)

                # get the logits'
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                    # evaluate the autoregressive loss at all positions
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            
            # reduce stats across all instances
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"Hellaswag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} hellaswag {acc_norm:.4f}\n")
            

        if (step > 0 and step % 250 == 0) and (not use_compile): # for torch.compile turn this off
            model.eval()
            num_return_sequences =  4
            max_length = 32

            # prefix tokens
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_length:
                # forward the model to get the logits
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(xgen) # (B, T, vocab_size)
                    # take the logits at the last position
                    logits = logits[:, -1, :] # (B, vocab_size)
                    # get the probabilities
                    probs = F.softmax(logits, dim=-1)
                    # do top-k sampling of 50 (huggingface pipeline default)
                    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    # select a token from the top-k probabilities
                    # note: multinomial does not demand the input to sum to 1
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                    # gather the corresponding indices
                    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                    # append to the sequence
                    xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.bfloat16): # allows activations to be in bfloat16, but params are left in float32
                logits, loss = model(x, y)
                loss = loss/grad_accum_steps # to normalize the loss as it would be in 1 batch
                loss_accum += loss.detach()
            if ddp: 
                DDP.require_backward_grad_sync = (micro_step == max_steps - 1) # only syncs gradients between gpus if done with it
            loss.backward() # this does += to gradients so is accumulating gradients over all iters
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine learning rate for this step
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize() # waits until gpu is done to get second time
        t1 = time.time()
        dt = (t1-t0)*1000 # time diff in miliseconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_second = tokens_processed / (t1-t0) # tokens per second throughput
        if master_process:
            print(f"step {step} | loss: {loss_accum.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f}")
            with open(log_file, 'a') as f:
                f.write(f"{step} train {loss_accum.item():.6f} \n")

    if ddp:
        destroy_process_group()


# run main if this is main process:
if __name__ == "__main__":
    main()