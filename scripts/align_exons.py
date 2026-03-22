"""
Exon alignment visualization script - extracts flanking sequences around exon boundaries
Finds exon start/end positions and outputs flanking sequence regions
"""
import os
import argparse
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Exon Alignment Tool")
print("=" * 60)

parser = argparse.ArgumentParser(description="Extract flanking sequences around exon boundaries")
parser.add_argument('--seq_file', type=str, default=os.path.abspath("../data/sequences.txt"), 
                    help='Sequence file (default: ../data/sequences.txt)')
parser.add_argument('--align_file', type=str, default=os.path.abspath("../data/alignments.txt"), 
                    help='Alignment/annotation file (default: ../data/alignments.txt)')
parser.add_argument('--out_loc', type=str, default=os.path.abspath("../data/aligned"), 
                    help='Output directory (default: ../data/aligned)')
parser.add_argument('--flank_size', type=int, default=40, 
                    help='Flanking size around exon boundaries (default: 40)')
parser.add_argument('--threshold', type=float, default=0.5, 
                    help='Threshold for annotation (default: 0.5)')
args = parser.parse_args()

print(f"Sequences file: {args.seq_file}")
print(f"Alignment file: {args.align_file}")
print(f"Output dir:     {args.out_loc}")
print(f"Flank size:     {args.flank_size}")
print(f"Threshold:      {args.threshold}")
print("=" * 60)

# Check if files exist, otherwise generate synthetic data
if not os.path.exists(args.seq_file) or not os.path.exists(args.align_file):
    if not os.path.exists(args.seq_file):
        print(f"Warning: Sequence file not found: {args.seq_file}")
    if not os.path.exists(args.align_file):
        print(f"Warning: Alignment file not found: {args.align_file}")
    
    print("Generating synthetic exon data for demonstration...")
    
    # Create data directory
    data_dir = os.path.dirname(args.seq_file)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic sequences
    num_seqs = 100
    seq_len = 200
    bases = ['A', 'C', 'G', 'T']
    
    sequences = []
    for _ in range(num_seqs):
        seq = ''.join(np.random.choice(bases, seq_len))
        sequences.append(seq)
    
    # Generate synthetic alignments (exon annotations)
    alignments = np.random.binomial(1, 0.3, (num_seqs, seq_len)).astype(float)
    
    # Save synthetic data
    with open(args.seq_file, 'w') as f:
        f.write('\n'.join(sequences))
    
    np.savetxt(args.align_file, alignments)
    
    print(f"✓ Generated {num_seqs} synthetic sequences")
    print(f"✓ Saved to {args.seq_file}")
    print(f"✓ Saved to {args.align_file}")
    print()

# Load data
try:
    with open(args.seq_file, "r") as f:
        seqs = np.vstack([np.expand_dims(np.array([c for c in line.strip()]), 0) for line in f.readlines()])
    
    a = np.loadtxt(args.align_file)
    
    print(f"Loaded {len(seqs)} sequences of length {seqs.shape[1]}")
    print()
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

# Find exon boundaries (transitions to/from high annotation values)
exon_starts = np.argmax(a >= args.threshold, 1)
exon_ends = -np.argmax((a >= args.threshold)[:, ::-1], 1)

# Extract flanking sequences
start_seqs = []
for idx, start in enumerate(exon_starts):
    if start > 0 and start < len(seqs[idx]):
        flank_start = max(0, start - args.flank_size // 2)
        flank_end = min(len(seqs[idx]), start + args.flank_size // 2)
        seq_slice = seqs[idx, flank_start:flank_end]
        start_seqs.append("".join(seq_slice))

end_seqs = []
for idx, end in enumerate(exon_ends):
    if end > 0 and end < len(seqs[idx]):
        flank_start = max(0, end - args.flank_size // 2)
        flank_end = min(len(seqs[idx]), end + args.flank_size // 2)
        seq_slice = seqs[idx, flank_start:flank_end]
        end_seqs.append("".join(seq_slice))

# Save results
os.makedirs(args.out_loc, exist_ok=True)

start_file = os.path.join(args.out_loc, "exon_starts_seqs.txt")
end_file = os.path.join(args.out_loc, "exon_ends_seqs.txt")

with open(start_file, "w") as f:
    f.write("\n".join(start_seqs))

with open(end_file, "w") as f:
    f.write("\n".join(end_seqs))

print(f"✓ Extracted {len(start_seqs)} exon start flanking sequences")
print(f"✓ Extracted {len(end_seqs)} exon end flanking sequences")
print(f"✓ Saved start sequences to {start_file}")
print(f"✓ Saved end sequences to {end_file}")
print("Done")