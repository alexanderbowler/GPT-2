import torch
from torch.nn import functional as F
from train_gpt2 import GPT2Config, GPT


device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
          )
device = 'cpu'
print(f"{device=}")

# running base gpt-2 model 124M params
checkpoint = torch.load("./log/model_19072.pt", map_location=device)
print(checkpoint['config'])
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])

num_return_sequences = 5
max_length = 60

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding('gpt2')

# get input and then generate potential outputs
print("This is the Alex's gpt2, please enter a phrase or sentence you wish to be completed:")
in_string = input()

tokens = enc.encode(in_string)
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5,8)
# print(tokens)
x = tokens.to(device)

# generate! x is (B,T) B=5 T=8
#set seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        # forward the model to get logits
        logits = model(x)[0]
        # get last logits for each batch
        logits = logits[:,-1,:] #(B, vocab_size)
        # get the probabilites via softmax
        probs = F.softmax(logits, dim=-1)
        # do topk sampling to get top 50 (default from HF)
        # topk_probs and topk_indices both (5,50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from topk
        ix = torch.multinomial(topk_probs, 1) # (B,1)
        # get corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

    
# print generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)