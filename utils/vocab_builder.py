import pandas as pd
import json
import argparse
from collections import Counter

def build_vocab(label_csv_path, vocab_output_path, min_freq=1):
    df = pd.read_csv(label_csv_path, sep='\t')  # Tab-separated file

    if 'SENTENCE_NAME' not in df.columns or 'SENTENCE' not in df.columns:
        print(f"🛑 Expected columns 'SENTENCE_NAME' and 'SENTENCE'. Found: {list(df.columns)}")
        return

    counter = Counter()
    for sentence in df['SENTENCE'].dropna():
        tokens = sentence.lower().strip().split()
        counter.update(tokens)

    vocab = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3
    }
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    with open(vocab_output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"✅ Vocabulary saved to {vocab_output_path} with {len(vocab)} tokens.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to label CSV (TSV format)')
    parser.add_argument('--output', type=str, default='vocab.json', help='Path to save vocab JSON')
    args = parser.parse_args()

    build_vocab(args.csv, args.output)
    print(f"✅ Vocabulary built and saved to {args.output}.")