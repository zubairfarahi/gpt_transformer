import torch
from models import GPTModel, create_dataloader_v1
from utils import get_tokenizer, generate_text_simple, token_ids_to_text, text_to_token_ids,generate, calc_loss_batch, calc_loss_loader
from config import load_config
import argparse
from train import train_model_simple
import matplotlib.pyplot as plt
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2("124M", "gpt2")

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig("losses.png")
    plt.show()


def train(args):

    GPT_CONFIG_124M = load_config(args.config_file)
    print("Loaded configuration:", GPT_CONFIG_124M)
    tokenizer = get_tokenizer("gpt2")
    
    file_path = "./data/the-verdict.txt"
    with open(file_path, "r", encoding='utf-8') as f:
        text_data = f.read()
    
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    
    print(total_characters, total_tokens)
    
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True
    )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False
    )
    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)
    
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs,
        eval_freq=5, eval_iter=1, start_context="Every effort moves you"
    )
    
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

import numpy as np
 
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
 
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
 
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])
 
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
 
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])
 
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
def prtrain(args):
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    # print(params["wte"])
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    GPT_CONFIG_124M = load_config(args.config_file)
    print("Loaded configuration:", GPT_CONFIG_124M)
    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias": True})
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()
    
    load_weights_into_gpt(gpt, params)
    
    
    tokenizer = get_tokenizer("gpt2")
    start_context = "Every effort moves you"
   
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(tokenizer, start_context), 
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    
def main(args):    
    print("Token embedding weight tensor dimensions:", params["wte"].shape)
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
    start_context = "Every effort moves you"
    # encoded = tokenizer.encode(start_context)
    # print("encoded:", encoded)
    # encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx=text_to_token_ids(tokenizer, start_context), 
        max_new_tokens=10, 
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))

    decoded_text = token_ids_to_text(tokenizer, out)
    print(decoded_text)
    


if __name__ == "__main__":
    # e.g: python3 main.py --config_file config/config.ini
    parser = argparse.ArgumentParser(description="Run GPT model with custom config")
    parser.add_argument('--config_file', type=str, help='Path to the configuration INI file', required=True)
    args = parser.parse_args()
    prtrain(args)
    # main(args)
    # train(args)
