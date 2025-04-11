import numpy as np

def pad_sequence(sequence, max_len, pad_value=0.0):
    length = sequence.shape[0]
    if length >= max_len:
        return sequence[:max_len]
    pad_shape = (max_len - length, sequence.shape[1])
    padding = np.full(pad_shape, pad_value, dtype=np.float32)
    return np.vstack((sequence, padding))

def normalize_keypoints(keypoints):
    # Normalize to [0, 1]
    min_val = np.min(keypoints, axis=0)
    max_val = np.max(keypoints, axis=0)
    return (keypoints - min_val) / (max_val - min_val + 1e-8)

def process_input(npz_data, max_len):
    hand = normalize_keypoints(npz_data['hand'])
    face = normalize_keypoints(npz_data['face'])
    body = normalize_keypoints(npz_data['body'])

    hand = pad_sequence(hand, max_len)
    face = pad_sequence(face, max_len)
    body = pad_sequence(body, max_len)

    return hand, face, body
