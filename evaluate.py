import torch
from torch.utils.data import DataLoader
from utils.data_loader import get_dataloader
from models.seq2seq import Seq2Seq
from models.hand_encoder import HandEncoder
from models.face_encoder import FaceEncoder
from models.body_encoder import BodyEncoder
from models.spatiotemporal_fusion import ModalityFusion
from models.nlp_decoder import NLPDecoder
from utils.utils import load_config, load_vocab, load_model
from utils.metrics import compute_bleu, compute_wer

def evaluate_model(model, dataloader, vocab, device):
    model.eval()
    references, hypotheses = [], []
    inv_vocab = {v: k for k, v in vocab.items()}
    
    with torch.no_grad():
        for src, trg, keypoint_lengths, _ in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, keypoint_lengths, teacher_forcing_ratio=0.0)
            preds = output.argmax(dim=-1).cpu().numpy()
            targets = trg.cpu().numpy()
            
            for pred_seq, target_seq in zip(preds, targets):
                pred_sentence = " ".join([inv_vocab.get(tok, "") for tok in pred_seq if tok not in {vocab["<sos>"], vocab["<eos>"], vocab["<pad>"]}])
                target_sentence = " ".join([inv_vocab.get(tok, "") for tok in target_seq if tok not in {vocab["<sos>"], vocab["<eos>"], vocab["<pad>"]}])
                hypotheses.append(pred_sentence)
                references.append(target_sentence)
    
    bleu = compute_bleu(references, hypotheses)
    wer_score = compute_wer(references, hypotheses)
    return bleu, wer_score

def main():
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab("vocab.json")
    
    test_loader = get_dataloader(config["test_data_path"], config["label_test_csv"], vocab, batch_size=1, shuffle=False)
    
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
    load_model(model, "models/best_model.pt")
    
    bleu, wer_score = evaluate_model(model, test_loader, vocab, device)
    print(f"BLEU Score: {bleu:.4f} | WER: {wer_score:.4f}")

if __name__ == "__main__":
    main()