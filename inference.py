import torch
import numpy as np
from models.seq2seq import Seq2Seq
from models.hand_encoder import HandEncoder
from models.face_encoder import FaceEncoder
from models.body_encoder import BodyEncoder
from models.spatiotemporal_fusion import ModalityFusion
from models.nlp_decoder import NLPDecoder
from models.t5_completion import T5Completer
from utils.utils import load_config, load_vocab, load_model
import cv2

class InferenceEngine:
    def __init__(self, model_path="models/best_model.pt"):
        self.config = load_config("config.yaml")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = load_vocab("vocab.json")
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.hand_encoder = HandEncoder(input_size=84, hidden_size=self.config["hand_encoder"]["hidden_size"],
                                        num_layers=self.config["hand_encoder"]["num_layers"],
                                        dropout=self.config["dropout"]).to(self.device)
        self.face_encoder = FaceEncoder(input_size=140, hidden_size=self.config["face_encoder"]["hidden_size"],
                                        num_layers=self.config["face_encoder"]["num_layers"],
                                        dropout=self.config["dropout"]).to(self.device)
        self.body_encoder = BodyEncoder(input_size=75, hidden_size=self.config["body_encoder"]["hidden_size"],
                                        num_layers=self.config["body_encoder"]["num_layers"],
                                        dropout=self.config["dropout"]).to(self.device)
        self.fusion = ModalityFusion(
            hand_dim=2*self.config["hand_encoder"]["hidden_size"],
            face_dim=2*self.config["face_encoder"]["hidden_size"],
            body_dim=2*self.config["body_encoder"]["hidden_size"],
            fused_dim=self.config["fusion"]["fused_dim"]
        ).to(self.device)
        self.decoder = NLPDecoder(
            vocab_size=self.config["vocab_size"],
            emb_dim=self.config["embedding_dim"],
            context_dim=self.config["fusion"]["fused_dim"],
            hidden_dim=self.config["decoder_hidden_dim"],
            dropout=self.config["dropout"]
        ).to(self.device)
        
        self.model = Seq2Seq(self.hand_encoder, self.face_encoder, self.body_encoder, self.fusion, self.decoder, self.device).to(self.device)
        load_model(self.model, model_path)
        self.model.eval()
        
        self.t5 = T5Completer(model_name=self.config["t5_model_name"], device=self.device) if self.config.get("use_t5_completion") else None
        
    def translate(self, keypoints: np.ndarray) -> str:
        with torch.no_grad():
            keypoints = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(self.device)
            keypoint_lengths = [keypoints.size(1)]  # Single sequence length
            output_tokens = self.model.translate(keypoints, max_len=self.config["max_translation_length"], keypoint_lengths=keypoint_lengths)
            output_tokens = output_tokens[0].cpu().numpy()
        
        words = [self.inv_vocab.get(tok, "") for tok in output_tokens if tok not in {self.vocab["<sos>"], self.vocab["<eos>"], self.vocab["<pad>"]}]
        sentence = " ".join(words)
        
        if self.t5:
            sentence = self.t5.correct(sentence)
        return sentence

if __name__ == "__main__":
    sample_npz = "data/processed/Test/_fZbAxSSbX4_0-5-rgb_front.npz"
    keypoints = np.load(sample_npz)['keypoints']
    engine = InferenceEngine()
    translation = engine.translate(keypoints)
    print("Translation:", translation)