import torch
from models import GPTModel, create_dataloader_v1
from utils import get_tokenizer, generate_text_simple, token_ids_to_text, text_to_token_ids, calc_loss_batch, calc_loss_loader
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
    main(args)
    # train(args)
