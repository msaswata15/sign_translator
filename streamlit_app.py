import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from models.tts import TextToSpeech
from inference import InferenceEngine
from utils.preprocess import preprocess_keypoints

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

engine = InferenceEngine()
tts = TextToSpeech()

st.title("🤟 Real-Time ASL Translator")
st.write("This application translates sign language continuously from your webcam.")

FRAME_WINDOW = st.image([])
TEXT_OUT = st.empty()

FRAME_THRESHOLD = engine.config["translation_threshold"]
cap = cv2.VideoCapture(0)
sequence = []

def extract_keypoints(results):
    keypoints = []
    
    if results.left_hand_landmarks:
        left_hand = [
            lm.x for lm in results.left_hand_landmarks.landmark
        ] + [
            lm.y for lm in results.left_hand_landmarks.landmark
        ]
        keypoints.extend(left_hand[:42])
    else:
        keypoints.extend([0.0] * 42)
    
    if results.right_hand_landmarks:
        right_hand = [
            lm.x for lm in results.right_hand_landmarks.landmark
        ] + [
            lm.y for lm in results.right_hand_landmarks.landmark
        ]
        keypoints.extend(right_hand[:42])
    else:
        keypoints.extend([0.0] * 42)
    
    if results.face_landmarks:
        face = [
            lm.x for lm in results.face_landmarks.landmark
        ] + [
            lm.y for lm in results.face_landmarks.landmark
        ]
        keypoints.extend(face[:140])
    else:
        keypoints.extend([0.0] * 140)
    
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark[:25]
        body = [
            lm.x for lm in pose_landmarks
        ] + [
            lm.y for lm in pose_landmarks
        ] + [
            lm.visibility for lm in pose_landmarks
        ]
        keypoints.extend(body[:75])
    else:
        keypoints.extend([0.0] * 75)
    
    return keypoints

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS
    )
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )
    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )

    keypoints = extract_keypoints(results)
    keypoints = preprocess_keypoints(keypoints, normalize=True, augment=False)
    
    if len(keypoints) == 299:
        sequence.append(keypoints)
    
    if len(sequence) >= FRAME_THRESHOLD:
        sequence_np = np.array(sequence)
        try:
            sentence = engine.translate(sequence_np)
            TEXT_OUT.markdown(f"**Prediction:** {sentence}")
            if engine.config.get("use_tts", False):
                tts.speak(sentence)
        except Exception as e:
            st.error(f"Inference error: {e}")
        sequence = []

    FRAME_WINDOW.image(frame, channels="BGR", use_column_width=True)
    
    if st.button("Stop"):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()