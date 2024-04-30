import torch
from models import GPTModel
from utils import get_tokenizer, generate_text_simple, decode_tokens
from config import load_config
import argparse

def main(args):
    GPT_CONFIG_124M = load_config(args.config_file)
    print("Loaded configuration:", GPT_CONFIG_124M)
    
    tokenizer = get_tokenizer("gpt2")
    txt = ["Every effort moves you", "Every day holds a"]
    batch = torch.stack([torch.tensor(tokenizer.encode(t)) for t in txt], dim=0)
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
    print(out)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    
    tokenizer = get_tokenizer("gpt2")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = decode_tokens(tokenizer, out.squeeze(0).tolist())
    print(decoded_text)
    


if __name__ == "__main__":
    # e.g: python3 main.py --config_file config/config.ini
    parser = argparse.ArgumentParser(description="Run GPT model with custom config")
    parser.add_argument('--config_file', type=str, help='Path to the configuration INI file', required=True)
    args = parser.parse_args()
    main(args)
