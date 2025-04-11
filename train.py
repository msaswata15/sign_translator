import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import yaml
from transformers import T5Tokenizer

from models.body_encoder import BodyEncoder
from models.face_encoder import FaceEncoder
from models.hand_encoder import HandEncoder
from models.spatio_temporal_decoder import SpatioTemporalDecoder
from models.seq2seq import Seq2Seq
from utils.data_loader import get_dataloaders
from utils.metrics import compute_bleu, compute_wer

# ==== Config ====
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==== Tokenizer ====
tokenizer = T5Tokenizer.from_pretrained("t5-small")
pad_idx = tokenizer.pad_token_id

# ==== Data ====
train_loader, val_loader, test_loader = get_dataloaders(
    config["data"]["npz_dir"],
    config["data"]["label_csv"],
    tokenizer,
    batch_size=config["train"]["batch_size"]
)

# ==== Models ====
hand_encoder = HandEncoder(config["model"]["hand_input"], config["model"]["hidden_dim"]).to(device)
face_encoder = FaceEncoder(config["model"]["face_input"], config["model"]["hidden_dim"]).to(device)
body_encoder = BodyEncoder(config["model"]["body_input"], config["model"]["hidden_dim"]).to(device)

decoder = SpatioTemporalDecoder(
    input_dim=config["model"]["hidden_dim"] * 3,
    hidden_dim=config["model"]["decoder_hidden"],
    output_dim=tokenizer.vocab_size
).to(device)

model = Seq2Seq(hand_encoder, face_encoder, body_encoder, decoder, device).to(device)

# ==== Optimizer & Loss ====
optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# ==== Training Loop ====
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for hand, face, body, target in tqdm(dataloader):
        hand, face, body, target = hand.to(device), face.to(device), body.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(hand, face, body, target)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        target = target[:, 1:].reshape(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ==== Validation ====
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for hand, face, body, target in dataloader:
            hand, face, body, target = hand.to(device), face.to(device), body.to(device), target.to(device)
            output = model(hand, face, body, target, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)

            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ==== Training ====
best_val_loss = float("inf")
save_path = config["train"]["save_path"]

for epoch in range(config["train"]["epochs"]):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{config['train']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to {save_path}")
