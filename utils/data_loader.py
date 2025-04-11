import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class SignDataset(Dataset):
    def __init__(self, npz_dir, label_csv, tokenizer, max_seq_len=100):
        self.npz_dir = npz_dir
        self.labels_df = pd.read_csv(label_csv)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = self.labels_df['filename'].tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_id = self.samples[idx]
        npz_path = os.path.join(self.npz_dir, file_id + ".npz")
        data = np.load(npz_path)

        hand = data['hand']    # (T, H)
        face = data['face']    # (T, F)
        body = data['body']    # (T, B)

        label = self.labels_df[self.labels_df['filename'] == file_id]['label'].values[0]

        tokens = self.tokenizer.encode(label, add_special_tokens=True, truncation=True, max_length=self.max_seq_len)
        tokens = torch.tensor(tokens)

        return torch.tensor(hand, dtype=torch.float32), \
               torch.tensor(face, dtype=torch.float32), \
               torch.tensor(body, dtype=torch.float32), \
               tokens

def get_dataloaders(npz_dir, label_csv, tokenizer, batch_size=4, split=(0.8, 0.1, 0.1)):
    dataset = SignDataset(npz_dir, label_csv, tokenizer)
    total_len = len(dataset)
    train_len = int(split[0] * total_len)
    val_len = int(split[1] * total_len)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )
