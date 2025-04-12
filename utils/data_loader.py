import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocess import preprocess_keypoints

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, label_csv, vocab, modalities=['hand', 'face', 'body']):
        self.data_dir = data_dir
        self.vocab = vocab
        self.modalities = modalities
        self.labels_df = pd.read_csv(label_csv, delimiter='\t')
        self.samples = self._load_samples()
        print(f"[INFO] Loaded {len(self.samples)} samples from {data_dir}")

    def _load_samples(self):
        samples = []
        for _, row in self.labels_df.iterrows():
            npz_path = os.path.join(self.data_dir, f"{row['SENTENCE_NAME']}.npz")
            if not os.path.exists(npz_path):
                print(f"[WARN] Missing .npz file: {npz_path}")
                continue
            try:
                keypoint_data = np.load(npz_path)
                keypoints = self._extract_modalities(keypoint_data)
                tokens = self._text_to_tokens(row['SENTENCE'])
                samples.append((keypoints, tokens))
            except Exception as e:
                print(f"[ERROR] Failed loading {npz_path}: {e}")
        return samples

    def _extract_modalities(self, data):
        all_keypoints = []
        keypoints = data['keypoints']  # Shape: (T, 299)
        if 'hand' in self.modalities:
            hand_kp = keypoints[:, :84]  # 42 left + 42 right
            all_keypoints.append(hand_kp)
        if 'face' in self.modalities:
            face_kp = keypoints[:, 84:224]  # 140
            all_keypoints.append(face_kp)
        if 'body' in self.modalities:
            body_kp = keypoints[:, 224:299]  # 75
            all_keypoints.append(body_kp)
        combined_keypoints = np.concatenate(all_keypoints, axis=-1)
        return combined_keypoints

    def _text_to_tokens(self, sentence):
        tokens = [self.vocab.get('<sos>')]
        tokens += [self.vocab.get(word.lower(), self.vocab.get('<unk>')) for word in sentence.strip().split()]
        tokens.append(self.vocab.get('<eos>'))
        return tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        keypoints, tokens = self.samples[idx]
        keypoints = preprocess_keypoints(keypoints, normalize=True, augment=True)
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    keypoints, tokens = zip(*batch)
    
    # Pad keypoint sequences
    keypoint_lengths = [kp.size(0) for kp in keypoints]
    max_keypoint_len = max(keypoint_lengths)
    padded_keypoints = torch.zeros(len(keypoints), max_keypoint_len, keypoints[0].size(1), dtype=torch.float32)
    for i, kp in enumerate(keypoints):
        padded_keypoints[i, :kp.size(0), :] = kp
    
    # Pad token sequences
    token_lengths = [len(t) for t in tokens]
    max_token_len = max(token_lengths)
    padded_tokens = torch.zeros(len(tokens), max_token_len, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded_tokens[i, :len(t)] = t
    
    return padded_keypoints, padded_tokens, keypoint_lengths, token_lengths

def get_dataloader(data_dir, label_csv, vocab, batch_size=4, shuffle=True):
    dataset = SignLanguageDataset(data_dir, label_csv, vocab)
    if len(dataset) == 0:
        raise ValueError(f"No samples loaded from {data_dir}. Please check your paths and .npz files.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)