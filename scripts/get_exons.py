import pandas as pd
import argparse
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import twobitreader as tb
except ImportError:
    print("Error: twobitreader package not installed")
    print("Install with: pip install twobitreader")
    sys.exit(1)

# Command-line arguments
parser = argparse.ArgumentParser(description='Extract exons from human genome')
parser.add_argument('--exons_file', type=str, 
                    help='Excel file with exon coordinates (knownCanonical.txt format)')
parser.add_argument('--genome_file', type=str,
                    help='2bit genome file (hg38.2bit format)')
parser.add_argument('--out_dir', type=str, default=os.path.abspath("../data"),
                    help='Output directory for extracted exons')
parser.add_argument('--min_size', type=int, default=50, help='Minimum exon size')
parser.add_argument('--max_size', type=int, default=400, help='Maximum exon size')
parser.add_argument('--window_size', type=int, default=500, help='Window size around exon')
args = parser.parse_args()

print("Extracting Exons from Genome")
print("=" * 50)
print(f"Exons file: {args.exons_file}")
print(f"Genome file: {args.genome_file}")
print(f"Output directory: {args.out_dir}")
print(f"Size range: {args.min_size} - {args.max_size}")
print(f"Window size: {args.window_size}")
print("=" * 50)

# Check if input files are provided
if args.exons_file is None or args.genome_file is None:
    print("Error: Both --exons_file and --genome_file are required")
    print()
    print("To download these files:")
    print("  1. UCSC Genes (knownCanonical.txt):")
    print("     http://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/knownCanonical.txt.gz")
    print()
    print("  2. UCSC Genome (hg38.2bit):")
    print("     http://hgdownload.cse.ucsc.edu/goldenpath/hg38/hg38.2bit")
    print()
    print("Example usage:")
    print("  python get_exons.py --exons_file knownCanonical.txt --genome_file hg38.2bit")
    sys.exit(1)

# Check if files exist
if not os.path.exists(args.exons_file):
    print(f"Error: Exons file not found: {args.exons_file}")
    sys.exit(1)

if not os.path.exists(args.genome_file):
    print(f"Error: Genome file not found: {args.genome_file}")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(args.out_dir, exist_ok=True)

print("\nLoading exon data...")
df = pd.read_csv(args.exons_file, skiprows=1, sep='\t')
print(f"✓ Loaded {len(df)} genes")

exon_chrs = []
exon_coords = np.empty([0, 2], np.int32)

for idx, row in df.iterrows():
    if (idx + 1) % 1000 == 0:
        print(f"  Processing gene {idx + 1}/{len(df)}...")
    
    new_chr = row["#hg38.knownCanonical.chrom"]
    new_starts = [int(string) for string in "".join(row["hg38.knownGene.exonStarts"]).split(",") if string]
    new_ends = [int(string) for string in "".join(row["hg38.knownGene.exonEnds"]).split(",") if string]
    
    assert len(new_starts) == len(new_ends)
    
    for idx_exon, _ in enumerate(new_starts):
        exon_chrs.append(new_chr)
        new_coords = np.array([new_starts[idx_exon], new_ends[idx_exon]]).reshape([1, 2])
        exon_coords = np.vstack([exon_coords, new_coords])

print(f"✓ Found {len(exon_chrs)} exons")

exon_size = exon_coords[:, 1] - exon_coords[:, 0]

filt_chrs = np.array(exon_chrs)[(exon_size >= args.min_size) & (exon_size <= args.max_size)]
filt_exons = exon_coords[(exon_size >= args.min_size) & (exon_size <= args.max_size)]

print(f"✓ Filtered to {len(filt_chrs)} exons in size range [{args.min_size}, {args.max_size}]")

print("\nExtracting sequences from genome...")
g = tb.TwoBitFile(args.genome_file)

file_strs = []
ann = np.empty([0, args.window_size], np.int32)

for idx, (chr, exon) in enumerate(zip(filt_chrs, filt_exons)):
    if (idx + 1) % 1000 == 0:
        print(f"  Extracted {idx} sequences...")
    
    try:
        c = g[chr]
        exon_start, exon_end = exon
        exon_len = exon_end - exon_start
        pad_size = args.window_size - exon_len
        left_pad = pad_size // 2
        right_pad = pad_size - left_pad
        str_ = c[exon_start - left_pad : exon_end + right_pad].upper()

        if 'N' not in str_ and len(str_) == args.window_size:
            file_strs.append(str_)

            base_ann = np.ones(args.window_size, np.int32)
            base_ann[:left_pad] = 0
            base_ann[left_pad + exon_len:] = 0
            ann = np.vstack([ann, np.expand_dims(base_ann, 0)])
    except Exception as e:
        # Skip chromosomes that might not exist in genome
        pass

print(f"✓ Successfully extracted {len(file_strs)} valid exon sequences")

# Save outputs
out_seqs_file = os.path.join(args.out_dir, 
                             f"exon_seqs_min{args.min_size}_max{args.max_size}_win{args.window_size}.txt")
out_ann_file = os.path.join(args.out_dir,
                            f"exon_ann_min{args.min_size}_max{args.max_size}_win{args.window_size}.txt")

with open(out_seqs_file, "w") as f:
    f.write("\n".join(file_strs))
print(f"✓ Saved sequences to {out_seqs_file}")

np.savetxt(out_ann_file, ann, fmt='%d')
print(f"✓ Saved annotations to {out_ann_file}")

print()
print("Done")
