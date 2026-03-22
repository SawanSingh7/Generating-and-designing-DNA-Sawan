"""
Convert binding data to training/validation/test splits
Processes binding affinity data and creates stratified dataset splits
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Binding Data Converter")
print("=" * 60)

parser = argparse.ArgumentParser(description="Convert binding data to train/valid/test splits")
parser.add_argument('--file', type=str, default=os.path.abspath("../data/binding_data.xlsx"), 
                    help='Excel file containing binding data (default: ../data/binding_data.xlsx)')
parser.add_argument('--output_dir', type=str, default=os.path.abspath("../data/binding_splits"), 
                    help='Output directory for splits (default: ../data/binding_splits)')
parser.add_argument('--valid_frac', type=float, default=0.15, 
                    help='Fraction of data to use as validation data (default: 0.15)')
parser.add_argument('--test_frac', type=float, default=0.15, 
                    help='Fraction of data to use as test data (default: 0.15)')
parser.add_argument('--column_name', type=str, default='Binding_Affinity', 
                    help='Which column to get scores from (default: Binding_Affinity)')
parser.add_argument('--seed', type=int, default=0, 
                    help='Random seed (default: 0)')
args = parser.parse_args()

print(f"Input file:      {args.file}")
print(f"Output dir:      {args.output_dir}")
print(f"Valid fraction:  {args.valid_frac}")
print(f"Test fraction:   {args.test_frac}")
print(f"Column name:     {args.column_name}")
print(f"Seed:            {args.seed}")
print("=" * 60)

# Check if file exists, otherwise generate synthetic data
if not os.path.exists(args.file):
    print(f"Warning: Binding data file not found: {args.file}")
    print("Generating synthetic binding data for demonstration...")
    
    # Create data directory
    data_dir = os.path.dirname(args.file)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic sequences and binding affinities
    num_samples = 500
    seq_len = 20
    bases = ['A', 'C', 'G', 'T']
    
    np.random.seed(args.seed)
    
    sequences = [''.join(np.random.choice(bases, seq_len)) for _ in range(num_samples)]
    
    # Generate binding affinities (varied distributions)
    binding_affinities = np.random.uniform(-5, 2, num_samples)
    
    # Create DataFrame with the requested column name
    df = pd.DataFrame({
        'Sequence': sequences,
        args.column_name: binding_affinities,
        'Type': np.random.choice(['Myc', 'Mad'], num_samples)
    })
    
    # Save to Excel
    df.to_excel(args.file, index=False, sheet_name='binding_data')
    
    print(f"✓ Generated {num_samples} synthetic binding data entries")
    print(f"✓ Saved to {args.file}")
    print()

# Load data
try:
    df = pd.read_excel(args.file)
    print(f"Loaded {len(df)} binding data entries")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# Check required columns
if 'Sequence' not in df.columns:
    print(f"Error: 'Sequence' column not found. Available: {list(df.columns)}")
    sys.exit(1)

if args.column_name not in df.columns:
    print(f"Error: Column '{args.column_name}' not found. Available: {list(df.columns)}")
    print(f"Using first available numeric column instead")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        args.column_name = numeric_cols[0]
        print(f"Using column: {args.column_name}")
    else:
        print("No numeric columns found for scores")
        sys.exit(1)

print()

# Shuffle and split
df = df.sample(frac=1, random_state=args.seed)
rows = len(df)

test_size = int(rows * args.test_frac) if args.test_frac > 0 else 0
valid_size = int(rows * args.valid_frac) if args.valid_frac > 0 else 0
train_size = rows - test_size - valid_size

train = df[:train_size]
valid = df[train_size : train_size + valid_size]
test = df[train_size + valid_size:]

# Create output location
save_loc = os.path.join(args.output_dir, args.column_name)
os.makedirs(save_loc, exist_ok=True)

# Save splits with both sequences and values
for name, subset in [("train", train), ("valid", valid), ("test", test)]:
    if len(subset) > 0:
        seq_file = os.path.join(save_loc, f"{name}_data.txt")
        val_file = os.path.join(save_loc, f"{name}_vals.txt")
        
        subset["Sequence"].to_csv(seq_file, index=False)
        subset[args.column_name].to_csv(val_file, index=False)

print(f"✓ Created splits:")
print(f"  Train: {len(train)} entries → {os.path.join(save_loc, 'train_data.txt')}")
print(f"  Valid: {len(valid)} entries → {os.path.join(save_loc, 'valid_data.txt')}")
print(f"  Test:  {len(test)} entries → {os.path.join(save_loc, 'test_data.txt')}")
print()

# Save metadata
metadata_file = os.path.join(save_loc, "metadata.txt")
with open(metadata_file, "w") as f:
    f.write(f"Total samples: {rows}\n")
    f.write(f"Train: {len(train)} ({len(train)/rows*100:.1f}%)\n")
    f.write(f"Valid: {len(valid)} ({len(valid)/rows*100:.1f}%)\n")
    f.write(f"Test:  {len(test)} ({len(test)/rows*100:.1f}%)\n")
    f.write(f"Score column: {args.column_name}\n")
    if len(subset) > 0:
        f.write(f"Score range: [{subset[args.column_name].min():.3f}, {subset[args.column_name].max():.3f}]\n")

print(f"✓ Saved metadata to {metadata_file}")
print("Done")
