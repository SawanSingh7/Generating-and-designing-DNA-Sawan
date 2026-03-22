"""
Dataset filtering script - keeps sequences based on value thresholds
Filters out sequences with values in a specified range
"""
import os
import argparse
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Dataset Chopping/Filtering Tool")
print("=" * 60)

parser = argparse.ArgumentParser(description="Filter DNA sequences by their associated values")
parser.add_argument('--sequences_file', type=str, default=os.path.abspath("../data/valid_data.txt"), 
                    help='Full filepath of sequences dataset (default: ../data/valid_data.txt)')
parser.add_argument('--values_file', type=str, default=os.path.abspath("../data/valid_vals.txt"), 
                    help='Full filepath of values dataset (default: ../data/valid_vals.txt)')
parser.add_argument('--out_loc', type=str, default=os.path.abspath("../data/chopped"), 
                    help='Where to save the chopped data (default: ../data/chopped)')
parser.add_argument('--name', type=str, default="valid", help="Name to use for new files")
parser.add_argument('--lower', type=float, default=0.3, help='Lower limit for removed data (default: 0.3)')
parser.add_argument('--upper', type=float, default=1.0, help='Upper limit for removed data (default: 1.0)')
parser.add_argument('--data_start', type=int, default=0, help='Number of rows to skip when loading data')
args = parser.parse_args()

print(f"Sequences file: {args.sequences_file}")
print(f"Values file:    {args.values_file}")
print(f"Output dir:     {args.out_loc}")
print(f"Filter range:   [{args.lower}, {args.upper}] (will REMOVE sequences in this range)")
print("=" * 60)

# Check if files exist, otherwise generate synthetic data
sequences_file_to_use = args.sequences_file
values_file_to_use = args.values_file

if not os.path.exists(sequences_file_to_use) or not os.path.exists(values_file_to_use):
    if not os.path.exists(sequences_file_to_use):
        print(f"Warning: Sequences file not found: {sequences_file_to_use}")
    if not os.path.exists(values_file_to_use):
        print(f"Warning: Values file not found: {values_file_to_use}")
    
    print("Generating synthetic data for demonstration...")
    
    # Create data directory - handle both filenames and paths
    data_dir = os.path.dirname(sequences_file_to_use)
    if not data_dir:
        # If just a filename was provided, use current directory
        data_dir = '.'
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Update paths to include directory
    sequences_file_to_use = os.path.join(data_dir, os.path.basename(sequences_file_to_use))
    values_file_to_use = os.path.join(data_dir, os.path.basename(values_file_to_use))
    
    num_seqs = 500
    seq_len = 36
    bases = ['A', 'C', 'G', 'T']
    
    sequences = []
    for _ in range(num_seqs):
        seq = ''.join(np.random.choice(bases, seq_len))
        sequences.append(seq)
    
    # Generate synthetic values (bimodal distribution for interesting filtering)
    values = np.concatenate([
        np.random.uniform(0.0, 0.2, num_seqs // 2),  # Low values
        np.random.uniform(0.8, 1.0, num_seqs // 2)   # High values
    ])
    np.random.shuffle(values)
    
    # Save synthetic data
    with open(sequences_file_to_use, 'w') as f:
        f.write('\n'.join(sequences))
    
    np.savetxt(values_file_to_use, values)
    
    print(f"✓ Generated {num_seqs} synthetic sequences")
    print(f"✓ Saved to {sequences_file_to_use}")
    print(f"✓ Saved to {values_file_to_use}")
    print()

# Load data
try:
    seqs = pd.read_csv(sequences_file_to_use, skiprows=args.data_start, header=None).squeeze()
    vals = pd.read_csv(values_file_to_use, skiprows=args.data_start, header=None).squeeze()
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

if len(seqs) != len(vals):
    print(f"Error: Number of sequences ({len(seqs)}) != number of values ({len(vals)})")
    sys.exit(1)

# Filter: keep sequences WITH values outside the range [lower, upper]
# i.e., keep values <= lower OR values >= upper
rows = (vals <= args.lower) | (vals >= args.upper)
num_kept = int(rows.sum())
num_removed = len(vals) - num_kept

print(f"Total sequences:     {len(vals)}")
print(f"Will REMOVE:         {len(vals) - num_kept} (with {args.lower} < value < {args.upper})")
print(f"Will KEEP:           {num_kept} (with value ≤ {args.lower} OR value ≥ {args.upper})")
print()

if num_kept == 0:
    print("Error: Did not find any datapoints in the specified range. Try different --lower/--upper values.")
    sys.exit(1)
else:
    chopped_seqs = seqs[rows]
    chopped_vals = vals[rows]
    
    # Save filtered data
    os.makedirs(args.out_loc, exist_ok=True)
    
    seqs_output = os.path.join(args.out_loc, f"{args.name}_data.txt")
    vals_output = os.path.join(args.out_loc, f"{args.name}_vals.txt")
    
    with open(seqs_output, "w") as f:
        f.write("\n".join(chopped_seqs.astype(str)))
    
    np.savetxt(vals_output, chopped_vals)
    
    print(f"✓ Saved {num_kept} sequences to {seqs_output}")
    print(f"✓ Saved {num_kept} values to {vals_output}")
    print()
    print(f"Value statistics for KEPT sequences:")
    print(f"  Mean:   {np.mean(chopped_vals):.6f}")
    print(f"  Median: {np.median(chopped_vals):.6f}")
    print(f"  Min:    {np.min(chopped_vals):.6f}")
    print(f"  Max:    {np.max(chopped_vals):.6f}")

print("Done")