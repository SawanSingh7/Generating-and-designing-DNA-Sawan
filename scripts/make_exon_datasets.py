import os
import sys
import numpy as np
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description='Create train/valid/test splits from exon data')
parser.add_argument('--data_loc', type=str, default=os.path.abspath("../data"),
                    help='Path for raw data (should contain exon_seqs_*.txt and exon_ann_*.txt files)')
parser.add_argument('--valid_frac', type=float, default=0.1, help='Percentage of raw data to use for validation set')
parser.add_argument('--test_frac', type=float, default=0.1, help='Percentage of raw data to use for test set')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

print("Creating Exon Datasets")
print("=" * 50)
print(f"Data location: {args.data_loc}")
print(f"Valid fraction: {args.valid_frac}")
print(f"Test fraction: {args.test_frac}")
print(f"Seed: {args.seed}")
print("=" * 50)

# Check if data directory exists
if not os.path.exists(args.data_loc):
    print(f"Error: Data directory not found: {args.data_loc}")
    sys.exit(1)

files = os.listdir(args.data_loc)
candidates = [file for file in files if "exon" in file]

if len(candidates) < 2:
    print(f"Warning: Only {len(candidates)} exon files found in {args.data_loc}")
    print(f"Need both exon_seqs_*.txt and exon_ann_*.txt files")
    print()
    print("To generate exon files, run:")
    print("  python get_exons.py --exons_file <path>/knownCanonical.txt --genome_file <path>/hg38.2bit")
    print()
    print("Generating synthetic exon data for testing...")
    
    # Generate sample exon sequences and annotations
    num_seqs = 200
    seq_len = 500
    bases = ['A', 'C', 'G', 'T']
    
    seqs = []
    ann = np.zeros((num_seqs, seq_len), dtype=np.int32)
    
    for i in range(num_seqs):
        # Generate random sequence
        seq = ''.join(np.random.choice(bases, seq_len))
        seqs.append(seq)
        
        # Random exon position within window
        exon_start = np.random.randint(100, 300)
        exon_end = np.random.randint(exon_start + 50, 400)
        ann[i, exon_start:exon_end] = 1
    
    # Save synthetic data
    seqs_file = os.path.join(args.data_loc, f"exon_seqs_synthetic.txt")
    ann_file = os.path.join(args.data_loc, f"exon_ann_synthetic.txt")
    
    with open(seqs_file, "w") as f:
        f.write("\n".join(seqs))
    
    np.savetxt(ann_file, ann, fmt='%d')
    
    print(f"✓ Generated {len(seqs)} synthetic exon sequences")
    candidates = [f for f in os.listdir(args.data_loc) if "exon" in f]
elif len(candidates) > 2:
    raise Exception("Too many candidate files in provided data_loc. There should be one sequence and one annotation file")

seqs_file = [file for file in candidates if "exon_seqs_" in file]
ann_file = [file for file in candidates if "exon_ann_" in file]

if seqs_file:
    seqs_file = seqs_file[0]
else:
    raise Exception("No sequence file recognized (looking for 'exon_seqs_' in filename)")

if ann_file:
    ann_file = ann_file[0]
else:
    raise Exception("No annotation file recognized (looking for 'exon_ann_' in filename)")

print(f"Loading sequence file: {seqs_file}")
print(f"Loading annotation file: {ann_file}")

with open(os.path.join(args.data_loc, seqs_file)) as f:
    seqs = [line.strip() for line in f.readlines()]

ann = np.loadtxt(os.path.join(args.data_loc, ann_file))
assert len(seqs) == len(ann), "Sequence and annotation files are incompatible shapes"

print(f"Loaded {len(seqs)} sequences")

np.random.seed(args.seed)
order = np.random.choice(len(seqs), size=len(seqs), replace=False)

seqs = np.array(seqs)[order]
ann = ann[order]

valid_size = int(args.valid_frac * len(seqs))
test_size = int(args.test_frac * len(seqs))

valid_seqs = seqs[:valid_size]
valid_ann = ann[:valid_size]
test_seqs = seqs[valid_size : valid_size + test_size]
test_ann = ann[valid_size : valid_size + test_size]
train_seqs = seqs[valid_size + test_size :]
train_ann = ann[valid_size + test_size :]

print(f"Train: {len(train_seqs)}, Valid: {len(valid_seqs)}, Test: {len(test_seqs)}")

# Save datasets
with open(os.path.join(args.data_loc, "train_data.txt"), "w") as f:
    f.write("\n".join(train_seqs))
print(f"✓ Saved train_data.txt ({len(train_seqs)} sequences)")

with open(os.path.join(args.data_loc, "valid_data.txt"), "w") as f:
    f.write("\n".join(valid_seqs))
print(f"✓ Saved valid_data.txt ({len(valid_seqs)} sequences)")

with open(os.path.join(args.data_loc, "test_data.txt"), "w") as f:
    f.write("\n".join(test_seqs))
print(f"✓ Saved test_data.txt ({len(test_seqs)} sequences)")

np.savetxt(os.path.join(args.data_loc, "train_ann.txt"), train_ann, fmt='%d')
print(f"✓ Saved train_ann.txt")

np.savetxt(os.path.join(args.data_loc, "valid_ann.txt"), valid_ann, fmt='%d')
print(f"✓ Saved valid_ann.txt")

np.savetxt(os.path.join(args.data_loc, "test_ann.txt"), test_ann, fmt='%d')
print(f"✓ Saved test_ann.txt")

print()
print("Done")