import tiktoken
import torch

# Function to get tokenizer instance
def get_tokenizer(model_name):
    
    return tiktoken.get_encoding(model_name)

# Function to encode text
def encode_text(tokenizer, text):

    return torch.tensor(tokenizer.encode(text))

# Function to decode tokens into text
def decode_tokens(tokenizer, tokens):

    return tokenizer.decode(tokens)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
       
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx