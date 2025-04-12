import yaml
import torch
import json

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_vocab(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model