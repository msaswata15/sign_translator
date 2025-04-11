import torch
from torch.nn.functional import softmax
from transformers import T5Tokenizer
import yaml
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer

from models.body_encoder import BodyEncoder
from models.face_encoder import FaceEncoder
from models.hand_encoder import HandEncoder
from models.spatio_temporal_decoder import SpatioTemporalDecoder
from models.seq2seq import Seq2Seq
from utils.data_loader import get_dataloaders

# ==== Config ====
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Tokenizer ====
tokenizer = T5Tokenizer.from_pretrained("t5-small")
pad_idx = tokenizer.pad_token_id

# ==== Load Data ====
_, _, test_loader = get_dataloaders(
    config["data"]["npz_dir"],
    config["data"]["label_csv"],
    tokenizer,
    batch_size=config["eval"]["batch_size"],
    mode="test"
)

# ==== Load Model ====
hand_encoder = HandEncoder(config["model"]["hand_input"], config["model"]["hidden_dim"]).to(device)
face_encoder = FaceEncoder(config["model"]["face_input"], config["model"]["hidden_dim"]).to(device)
body_encoder = BodyEncoder(config["model"]["body_input"], config["model"]["hidden_dim"]).to(device)

decoder = SpatioTemporalDecoder(
    input_dim=config["model"]["hidden_dim"] * 3,
    hidden_dim=config["model"]["decoder_hidden"],
    output_dim=tokenizer.vocab_size
).to(device)

model = Seq2Seq(hand_encoder, face_encoder, body_encoder, decoder, device).to(device)
model.load_state_dict(torch.load(config["train"]["save_path"], map_location=device))
model.eval()

# ==== Inference ====
def decode_sequence(output_logits):
    tokens = output_logits.argmax(dim=-1).tolist()
    decoded_sentences = []
    for seq in tokens:
        # Remove padding and stop at </s>
        words = tokenizer.decode(seq, skip_special_tokens=True)
        decoded_sentences.append(words)
    return decoded_sentences

def evaluate_bleu_wer(model, dataloader):
    all_bleu_scores = []
    all_wers = []

    with torch.no_grad():
        for hand, face, body, targets in tqdm(dataloader):
            hand, face, body = hand.to(device), face.to(device), body.to(device)
            outputs = model(hand, face, body, targets, teacher_forcing_ratio=0.0)

            predictions = decode_sequence(outputs)
            references = decode_sequence(targets)

            for pred, ref in zip(predictions, references):
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                bleu = sentence_bleu([ref_tokens], pred_tokens)
                error = wer(ref, pred)
                all_bleu_scores.append(bleu)
                all_wers.append(error)

    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
    avg_wer = sum(all_wers) / len(all_wers)
    return avg_bleu, avg_wer

# ==== Run Evaluation ====
bleu_score, wer_score = evaluate_bleu_wer(model, test_loader)
print(f"\n✅ BLEU Score: {bleu_score:.4f}")
print(f"❌ WER Score : {wer_score:.4f}")
