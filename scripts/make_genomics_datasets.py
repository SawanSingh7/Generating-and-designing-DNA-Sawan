"""
Genomic dataset splitter - creates train/valid/test splits from genomic sequences
Filters sequences and creates stratified splits for model training
"""
import os
import sys
import numpy as np
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Genomics Dataset Splitter")
print("=" * 60)

parser = argparse.ArgumentParser(description="Create train/valid/test splits from genomic sequences")
parser.add_argument('--data_loc', type=str, default=os.path.abspath("../data/genomics_raw.txt"), 
                    help='Location of raw data file (default: ../data/genomics_raw.txt)')
parser.add_argument('--valid_frac', type=float, default=0.1, 
                    help='Fraction of data for validation set (default: 0.1)')
parser.add_argument('--test_frac', type=float, default=0.1, 
                    help='Fraction of data for test set (default: 0.1)')
parser.add_argument('--remove_N', action='store_true', 
                    help='Remove sequences with unknown characters (default: keep all)')
parser.add_argument('--seed', type=int, default=None, 
                    help='Random seed for reproducibility')
args = parser.parse_args()

print(f"Data file:     {args.data_loc}")
print(f"Valid fraction: {args.valid_frac}")
print(f"Test fraction:  {args.test_frac}")
print(f"Filter N chars: {args.remove_N}")
print(f"Seed:          {args.seed}")
print("=" * 60)

# Expand path
data_file = os.path.expanduser(args.data_loc)
data_path = os.path.split(data_file)[0]

# Check if file exists, otherwise generate synthetic data
if not os.path.exists(data_file):
    print(f"Warning: Data file not found: {data_file}")
    print("Generating synthetic genomic sequences for demonstration...")
    
    # Create data directory
    os.makedirs(data_path, exist_ok=True)
    
    # Generate synthetic genomic sequences
    num_seqs = 500
    seq_len = 100
    bases = ['A', 'C', 'G', 'T']
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    sequences = []
    for _ in range(num_seqs):
        # Generate mostly clean sequences, some with N's
        if np.random.random() < 0.1:  # 10% with N characters
            seq = ''.join(np.random.choice(bases + ['N'], seq_len))
        else:
            seq = ''.join(np.random.choice(bases, seq_len))
        sequences.append(seq)
    
    # Save synthetic data
    with open(data_file, 'w') as f:
        f.write('\n'.join(sequences))
    
    print(f"✓ Generated {num_seqs} synthetic sequences")
    print(f"✓ Saved to {data_file}")
    print()

# Load data
try:
    with open(data_file) as f:
        if args.remove_N:
            seqs = [line.strip().upper() for line in f.readlines() if 'N' not in line.upper()]
        else:
            seqs = [line.strip().upper() for line in f.readlines()]
    
    print(f"Loaded {len(seqs)} sequences")
    if args.remove_N:
        with open(data_file) as f:
            total = len(f.readlines())
        print(f"Filtered out {total - len(seqs)} sequences with N characters")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

if len(seqs) == 0:
    print("Error: No sequences loaded")
    sys.exit(1)

# Shuffle
np.random.seed(args.seed)
order = np.random.choice(len(seqs), size=len(seqs), replace=False)
seqs = np.array(seqs)[order]

# Split
valid_size = int(args.valid_frac * len(seqs))
test_size = int(args.test_frac * len(seqs))
train_size = len(seqs) - valid_size - test_size

valid_seqs = seqs[:valid_size]
test_seqs = seqs[valid_size : valid_size + test_size]
train_seqs = seqs[valid_size + test_size :]

# Ensure output directory exists
os.makedirs(data_path, exist_ok=True)

# Save splits
train_file = os.path.join(data_path, "train_data.txt")
valid_file = os.path.join(data_path, "valid_data.txt")
test_file = os.path.join(data_path, "test_data.txt")

with open(train_file, "w") as f:
    f.write("\n".join(train_seqs))
with open(valid_file, "w") as f:
    f.write("\n".join(valid_seqs))
with open(test_file, "w") as f:
    f.write("\n".join(test_seqs))

print()
print(f"✓ Created splits:")
print(f"  Train: {len(train_seqs)} sequences ({len(train_seqs)/len(seqs)*100:.1f}%) → {train_file}")
print(f"  Valid: {len(valid_seqs)} sequences ({len(valid_seqs)/len(seqs)*100:.1f}%) → {valid_file}")
print(f"  Test:  {len(test_seqs)} sequences ({len(test_seqs)/len(seqs)*100:.1f}%) → {test_file}")
print("Done")
