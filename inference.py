import cv2
import torch
import numpy as np
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pyttsx3

from models.body_encoder import BodyEncoder
from models.face_encoder import FaceEncoder
from models.hand_encoder import HandEncoder
from models.spatio_temporal_decoder import SpatioTemporalDecoder
from models.seq2seq import Seq2Seq
from utils.preprocess import extract_keypoints_from_frame
from utils.metrics import post_process_text

# ========== Config ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 20  # number of frames to consider for one prediction window

# ========== Load Models ==========
tokenizer = T5Tokenizer.from_pretrained("t5-small")

hand_encoder = HandEncoder(input_dim=63, hidden_dim=128).to(DEVICE)
face_encoder = FaceEncoder(input_dim=70, hidden_dim=128).to(DEVICE)
body_encoder = BodyEncoder(input_dim=75, hidden_dim=128).to(DEVICE)
decoder = SpatioTemporalDecoder(input_dim=384, hidden_dim=256, output_dim=tokenizer.vocab_size).to(DEVICE)

model = Seq2Seq(hand_encoder, face_encoder, body_encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(torch.load("models/final_model.pth", map_location=DEVICE))
model.eval()

# T5 for sentence completion
t5 = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Text-to-speech
engine = pyttsx3.init()

# ========== Streamlit App ==========
st.title("🤟 Real-time Sign Language Translator")
st.text("Perform a gesture in front of the webcam. It will continuously translate to text and speech.")

frame_window = st.image([])
text_output = st.empty()

cap = cv2.VideoCapture(0)

frames = []

def translate_keypoints_sequence(seq_hand, seq_face, seq_body):
    with torch.no_grad():
        hand_tensor = torch.tensor(seq_hand, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        face_tensor = torch.tensor(seq_face, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        body_tensor = torch.tensor(seq_body, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        input_ids = torch.full((1, 10), tokenizer.pad_token_id).to(DEVICE)
        input_ids[:, 0] = tokenizer.pad_token_id  # <sos>

        outputs = model(hand_tensor, face_tensor, body_tensor, input_ids, teacher_forcing_ratio=0.0)
        pred_tokens = outputs.argmax(dim=-1).squeeze(0).tolist()
        translated_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        return translated_text

def complete_sentence(text):
    input_text = f"complete: {text}"
    input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)
    outputs = t5.generate(input_ids, max_length=40, num_beams=2, early_stopping=True)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def speak(text):
    engine.say(text)
    engine.runAndWait()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        keypoints = extract_keypoints_from_frame(frame)

        if keypoints is not None:
            hand_kp, face_kp, body_kp = keypoints
            frames.append((hand_kp, face_kp, body_kp))

        if len(frames) >= SEQ_LEN:
            seq_hand = [f[0] for f in frames[-SEQ_LEN:]]
            seq_face = [f[1] for f in frames[-SEQ_LEN:]]
            seq_body = [f[2] for f in frames[-SEQ_LEN:]]

            raw_translation = translate_keypoints_sequence(seq_hand, seq_face, seq_body)
            completed_sentence = complete_sentence(raw_translation)
            completed_sentence = post_process_text(completed_sentence)

            text_output.markdown(f"**✍️ Detected Sentence:** {completed_sentence}")
            speak(completed_sentence)

            frames = []  # clear for next window

        frame_window.image(frame, channels="BGR")

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
