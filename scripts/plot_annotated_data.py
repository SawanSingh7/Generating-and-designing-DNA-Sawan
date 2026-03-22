"""
Annotated sequence visualizer - creates markup showing annotations on sequences
Displays annotations as uppercase/lowercase markup in sequence output
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Annotated Sequence Visualizer")
print("=" * 60)

parser = argparse.ArgumentParser(description="Visualize annotations on DNA sequences")
parser.add_argument('--seqs', type=str, default=os.path.abspath("../data/annotated_seqs.txt"), 
                    help="File containing sequences (default: ../data/annotated_seqs.txt)")
parser.add_argument('--ann', type=str, default=os.path.abspath("../data/annotations.txt"), 
                    help="File containing annotations (default: ../data/annotations.txt)")
parser.add_argument('--out', type=str, default=os.path.abspath("../data/annotated_output.txt"), 
                    help="Filepath for final output (default: ../data/annotated_output.txt)")
parser.add_argument('--num_seqs', type=int, default=None, 
                    help="Number of sequences to use for final output (default: all)")
args = parser.parse_args()

print(f"Sequences:  {args.seqs}")
print(f"Annotations: {args.ann}")
print(f"Output:     {args.out}")
print("=" * 60)

# Check if files exist, otherwise generate synthetic data
print()
if not os.path.exists(args.seqs) or not os.path.exists(args.ann):
    print("Warning: Input files not found")
    print("Generating synthetic annotated data for demonstration...")
    
    # Create data directory
    data_dir = os.path.dirname(args.seqs)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic sequences and annotations
    num_seqs = 50
    seq_len = 100
    bases = ['A', 'C', 'G', 'T']
    
    # Generate sequences
    sequences = []
    for _ in range(num_seqs):
        seq = ''.join(np.random.choice(bases, seq_len))
        sequences.append(seq)
    
    # Generate binary annotations (0 or 1, representing annotated regions)
    annotations = np.random.binomial(1, 0.3, (num_seqs, seq_len)).astype(float)
    
    # Save synthetic data
    with open(args.seqs, 'w') as f:
        f.write('\n'.join(sequences))
    
    np.savetxt(args.ann, annotations)
    
    print(f"✓ Generated {num_seqs} synthetic sequences ({seq_len}bp)")
    print(f"✓ Generated {num_seqs} annotation tracks")
    print(f"✓ Saved sequences to {args.seqs}")
    print(f"✓ Saved annotations to {args.ann}")
    print()

# Load data
try:
    with open(os.path.expanduser(args.seqs), "r") as f:
        s = [line.strip() for line in f.readlines() if line.strip()]
    a = np.loadtxt(os.path.expanduser(args.ann))
    
    # Handle single annotation row (when there's only 1 sequence)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    
    print(f"Loaded {len(s)} sequences")
    print(f"Loaded annotations: shape {a.shape}")
    
    assert len(s) == a.shape[0], "Number of sequences and annotations must match."
    print()
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

def inline_annotate(sequence, annotation):
    """Create markup showing annotations as uppercase/lowercase"""
    out_seq = ""
    for char, val in zip(sequence, annotation):
        out_seq += char.upper() if val >= 0.5 else char.lower()
    return out_seq

# Apply annotations
ann_seqs = [inline_annotate(seq, ann) for seq, ann in zip(s, a)]

# Limit to num_seqs if specified
if args.num_seqs is not None:
    ann_seqs = ann_seqs[:args.num_seqs]

# Ensure output directory exists
os.makedirs(os.path.dirname(os.path.expanduser(args.out)), exist_ok=True)

# Write output
with open(os.path.expanduser(args.out), "w") as f:
    f.write("\n".join(ann_seqs))

print(f"✓ Generated {len(ann_seqs)} annotated sequences")
print(f"✓ Saved to {args.out}")
print(f"  (Uppercase = annotated region, lowercase = unannotated)")
print("Done")