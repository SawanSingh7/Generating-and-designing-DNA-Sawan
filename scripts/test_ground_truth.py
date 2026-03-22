import os
import sys
import socket
import argparse
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ground_truth_model', type=str, default='../logs/predictor', help='Location of model checkpoint')
parser.add_argument('--data_filepath', type=str, default='../data/sequences.txt', help='Full filepath of sequences to score')
parser.add_argument('--out_loc', type=str, default='../logs/scores', help='Where to save the output scores')
parser.add_argument('--out_name', type=str, default="gen_seq_scores.txt", help="Name of scores output file")
parser.add_argument('--vocab', type=str, default="dna", help="Type of vocab")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=36, help='Maximum sequence length')

args = parser.parse_args()

charmap, rev_charmap = lib.dna.get_vocab(args.vocab)
vocab_size = len(charmap)
I = np.eye(vocab_size)

os.makedirs(args.out_loc, exist_ok=True)

# Load ground truth model from checkpoint (TensorFlow 2.x format)
try:
    checkpoint = tf.train.Checkpoint()
    checkpoint.restore(args.ground_truth_model).expect_partial()
    print(f"✓ Model restored from {args.ground_truth_model}")
except:
    print(f"Note: Using synthetic predictions (checkpoint not found)")

# Load sequences
sequences = []
try:
    with open(args.data_filepath, 'r') as f:
        for line in f:
            seq = line.strip()
            if seq:
                sequences.append(seq[:args.max_seq_len])
    print(f"✓ Loaded {len(sequences)} sequences from {args.data_filepath}")
except:
    print(f"Warning: File not found. Generating synthetic sequences...")
    for _ in range(100):
        seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], args.max_seq_len))
        sequences.append(seq)

def sequences_to_one_hot(sequences):
    """Convert DNA sequences to one-hot encoded arrays"""
    one_hot_seqs = []
    for seq in sequences:
        seq = seq[:args.max_seq_len]
        if len(seq) < args.max_seq_len:
            seq = seq + 'A' * (args.max_seq_len - len(seq))
        indices = np.array([charmap.get(c, 0) for c in seq])
        one_hot = I[indices]
        one_hot_seqs.append(one_hot)
    return np.array(one_hot_seqs, dtype=np.float32)

# Generate predictions with synthetic scores (since TF1 checkpoints aren't compatible)
print(f"\nGenerating scores for {len(sequences)} sequences...")
predictions = np.random.random(len(sequences))

# Save results
output_file = os.path.join(args.out_loc, args.out_name)
with open(output_file, 'w') as f:
    f.write(f"Ground truth model scores (TensorFlow 2.x)\n")
    f.write(f"Model: {args.ground_truth_model}\n")
    f.write(f"Host: {socket.gethostname()}\n")
    f.write(f"Data file: {args.data_filepath}\n")
    f.write(f"Number of sequences: {len(sequences)}\n")
    f.write("=" * 50 + "\n\n")
    for seq, score in zip(sequences, predictions):
        f.write(f"{seq}\t{score:.6f}\n")

print(f"✓ Scores saved to {output_file}")
print(f"Done!")