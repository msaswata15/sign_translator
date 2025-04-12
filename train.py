import torch
import torch.nn as nn
from utils.data_loader import get_dataloader
from models.seq2seq import Seq2Seq
from models.hand_encoder import HandEncoder
from models.face_encoder import FaceEncoder
from models.body_encoder import BodyEncoder
from models.spatiotemporal_fusion import ModalityFusion
from models.nlp_decoder import NLPDecoder
from utils.utils import load_config, load_vocab, save_model
import matplotlib.pyplot as plt

def train(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    total_loss = 0
    for src, trg, keypoint_lengths, _ in dataloader:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg, keypoint_lengths)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg, keypoint_lengths, _ in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, keypoint_lengths, teacher_forcing_ratio=0.0)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def plot_losses(train_losses, val_losses, save_path="loss_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig(save_path)
    plt.close()

def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab = load_vocab("vocab.json")
    
    train_loader = get_dataloader(config["train_data_path"], config["label_train_csv"], vocab, 
                                  batch_size=config["batch_size"], shuffle=True)
    val_loader = get_dataloader(config["val_data_path"], config["label_val_csv"], vocab, 
                                batch_size=config["batch_size"], shuffle=False)
    
    hand_encoder = HandEncoder(input_size=84, hidden_size=config["hand_encoder"]["hidden_size"],
                               num_layers=config["hand_encoder"]["num_layers"],
                               dropout=config["dropout"])
    face_encoder = FaceEncoder(input_size=140, hidden_size=config["face_encoder"]["hidden_size"],
                               num_layers=config["face_encoder"]["num_layers"],
                               dropout=config["dropout"])
    body_encoder = BodyEncoder(input_size=75, hidden_size=config["body_encoder"]["hidden_size"],
                               num_layers=config["body_encoder"]["num_layers"],
                               dropout=config["dropout"])
    fusion = ModalityFusion(hand_dim=2*config["hand_encoder"]["hidden_size"],
                            face_dim=2*config["face_encoder"]["hidden_size"],
                            body_dim=2*config["body_encoder"]["hidden_size"],
                            fused_dim=config["fusion"]["fused_dim"])
    decoder = NLPDecoder(vocab_size=config["vocab_size"],
                         emb_dim=config["embedding_dim"],
                         context_dim=config["fusion"]["fused_dim"],
                         hidden_dim=config["decoder_hidden_dim"],
                         dropout=config["dropout"])
    
    model = Seq2Seq(hand_encoder, face_encoder, body_encoder, fusion, decoder, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=config["pad_token"])
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(config["epochs"]):
        train_loss = train(model, train_loader, optimizer, criterion, config["clip"], device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "models/best_model.pt")
            print("[INFO] Saved best model.")
    
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()