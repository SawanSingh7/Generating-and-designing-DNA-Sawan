import heapq
import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import editdistance as ed
except ImportError:
    print("Error: editdistance package not installed")
    print("Install with: pip install editdistance")
    sys.exit(1)

# Command-line arguments
parser = argparse.ArgumentParser(description='Calculate edit distance between sequences')
parser.add_argument('--seq_file', type=str, help='Input sequences file to analyze')
parser.add_argument('--train_file', type=str, default=os.path.abspath("../logs/sim_ground_truth_data/train_data.txt"),
                    help='Training sequences file for reference')
parser.add_argument('--out_file', type=str, help='Output file for edit distance results')
parser.add_argument('--batch_size', type=int, default=250, help='Batch size for processing')
parser.add_argument('--set', type=str, default='test', choices=['gen', 'test', 'train'],
                    help='Dataset type (gen=generated, test=test, train=training)')
args = parser.parse_args()

# Set defaults based on dataset type if not explicitly provided
if args.seq_file is None:
    if args.set == "gen":
        args.seq_file = os.path.abspath("../logs/pp_test/samples/samples_100000.txt")
    elif args.set == "test":
        args.seq_file = os.path.abspath("../logs/sim_ground_truth_data/test_data.txt")
    elif args.set == "train":
        args.seq_file = os.path.abspath("../logs/sim_ground_truth_data/train_data.txt")

if args.out_file is None:
    base_dir = os.path.abspath("../logs/edit_distance")
    os.makedirs(base_dir, exist_ok=True)
    args.out_file = os.path.join(base_dir, f"{args.set}_edit_distance.txt")

print(f"Edit Distance Calculator")
print(f"=" * 50)
print(f"Input sequences: {args.seq_file}")
print(f"Training (reference) sequences: {args.train_file}")
print(f"Output file: {args.out_file}")
print(f"Batch size: {args.batch_size}")
print(f"=" * 50)

def generate_dna_sequences(num_seq, length=36):
    """Generate random DNA sequences"""
    bases = ['A', 'C', 'G', 'T']
    seqs = []
    for _ in range(num_seq):
        seq = ''.join(np.random.choice(bases, length))
        seqs.append(seq)
    return seqs

# Check if files exist, generate if needed
if not os.path.exists(args.seq_file):
    print(f"Warning: Sequence file not found: {args.seq_file}")
    print(f"Generating synthetic sequences...")
    os.makedirs(os.path.dirname(args.seq_file), exist_ok=True)
    seqs = generate_dna_sequences(100)
    with open(args.seq_file, "w") as f:
        f.write("\n".join(seqs))
    print(f"✓ Generated {len(seqs)} sequences to {args.seq_file}")
else:
    print(f"✓ Loading sequences from {args.seq_file}")
    with open(args.seq_file, "r") as f:
        seqs = [seq.strip() for seq in f.readlines()]

if not os.path.exists(args.train_file):
    print(f"Warning: Training file not found: {args.train_file}")
    print(f"Generating synthetic training sequences...")
    os.makedirs(os.path.dirname(args.train_file), exist_ok=True)
    train_seqs = generate_dna_sequences(50)
    with open(args.train_file, "w") as f:
        f.write("\n".join(train_seqs))
    print(f"✓ Generated {len(train_seqs)} training sequences to {args.train_file}")
else:
    print(f"✓ Loading training sequences from {args.train_file}")
    with open(args.train_file, "r") as f:
        train_seqs = [seq.strip() for seq in f.readlines()]

print(f"Loaded {len(seqs)} sequences to analyze")
print(f"Using {len(train_seqs)} reference sequences")
print()

def min_edit_dist(gen_seq):
    """Calculate minimum edit distance to training set"""
    h = []
    for train_seq in train_seqs:
        d = ed.eval(gen_seq, train_seq)
        heapq.heappush(h, d)
    if args.set == "train":
        # For training set, return second smallest (to skip identical)
        return heapq.nsmallest(2, h)[-1] if len(h) > 1 else h[0]
    else:
        return heapq.nsmallest(1, h)[0]

# Ensure output directory exists
os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

# Clear output file
with open(args.out_file, "w") as f:
    f.write(f"# Edit distance results for {args.set} set\n")
    f.write(f"# Columns: sequence, min_edit_distance\n")

# Process sequences in batches
num_seqs = len(seqs)
batch_size = args.batch_size
if batch_size < num_seqs:
    num_batches = (num_seqs + batch_size - 1) // batch_size
else:
    batch_size = num_seqs
    num_batches = 1

for idx in range(num_batches):
    batch = seqs[idx * batch_size : (idx + 1) * batch_size]
    d = [[g, min_edit_dist(g)] for g in batch]

    with open(args.out_file, "a") as f:
        out_str = "\n".join("{} {}".format(tuple_[0], tuple_[1]) for tuple_ in d) + "\n"
        f.write(out_str)
    
    num_processed = min((idx + 1) * batch_size, num_seqs)
    print(f"Processed {num_processed} of {num_seqs} sequences ({100*num_processed/num_seqs:.1f}%)")

print()
print(f"✓ Results saved to {args.out_file}")
print("Done")