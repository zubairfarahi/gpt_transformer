import configparser

def load_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    if 'GPTModel' not in config:
        raise ValueError("Section 'GPTModel' not found in the configuration file.")

    gpt_config = {
        "vocab_size": int(config['GPTModel']['vocab_size']),
        "context_length": int(config['GPTModel']['context_length']),
        "emb_dim": int(config['GPTModel']['emb_dim']),
        "n_heads": int(config['GPTModel']['n_heads']),
        "n_layers": int(config['GPTModel']['n_layers']),
        "drop_rate": float(config['GPTModel']['drop_rate']),
        "qkv_bias": config['GPTModel']['qkv_bias'].lower() in ['true', '1', 't', 'y', 'yes']
    }
    return gpt_config
