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
        self.samples = self.labels_df[['SENTENCE_NAME', 'SENTENCE']].values.tolist()
        print(f"[INFO] Found {len(self.samples)} sample entries from {data_dir}")

    def _extract_modalities(self, keypoints):
        all_keypoints = []
        if 'hand' in self.modalities:
            hand_kp = keypoints[:, :84]
            all_keypoints.append(hand_kp)
        if 'face' in self.modalities:
            face_kp = keypoints[:, 84:224]
            all_keypoints.append(face_kp)
        if 'body' in self.modalities:
            body_kp = keypoints[:, 224:299]
            all_keypoints.append(body_kp)
        return np.concatenate(all_keypoints, axis=-1)

    def _text_to_tokens(self, sentence):
        tokens = [self.vocab.get('<sos>')]
        tokens += [self.vocab.get(word.lower(), self.vocab.get('<unk>')) for word in sentence.strip().split()]
        tokens.append(self.vocab.get('<eos>'))
        return tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence_name, sentence = self.samples[idx]
        npz_path = os.path.join(self.data_dir, f"{sentence_name}.npz")
        
        try:
            data = np.load(npz_path)
            keypoints = self._extract_modalities(data['keypoints'])
        except Exception as e:
            print(f"[ERROR] Could not load {npz_path}: {e}")
            keypoints = np.zeros((1, 299))  # Fallback to avoid crash

        tokens = self._text_to_tokens(sentence)
        keypoints = preprocess_keypoints(keypoints, normalize=True, augment=True)
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    keypoints, tokens = zip(*batch)

    keypoint_lengths = [kp.size(0) for kp in keypoints]
    max_kp_len = max(keypoint_lengths)
    padded_kp = torch.zeros(len(keypoints), max_kp_len, keypoints[0].size(1))
    for i, kp in enumerate(keypoints):
        padded_kp[i, :kp.size(0)] = kp

    token_lengths = [len(t) for t in tokens]
    max_tok_len = max(token_lengths)
    padded_tokens = torch.zeros(len(tokens), max_tok_len, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded_tokens[i, :len(t)] = t

    return padded_kp, padded_tokens, keypoint_lengths, token_lengths

def get_dataloader(data_dir, label_csv, vocab, batch_size=4, shuffle=True):
    dataset = SignLanguageDataset(data_dir, label_csv, vocab)
    if len(dataset) == 0:
        raise ValueError(f"No samples loaded from {data_dir}. Please check your paths and .npz files.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
