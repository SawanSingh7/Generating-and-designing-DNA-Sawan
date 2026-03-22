"""
Dataset cutoff computation script - calculates percentile thresholds for binding values
Computes cutoff values across experiment directories for classification tasks
"""
import os
import argparse
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Dataset Cutoff Computation Tool")
print("=" * 60)

parser = argparse.ArgumentParser(description="Compute cutoff thresholds for binding experiments")
parser.add_argument('--base_path', type=str, default=os.path.abspath("../data"), 
                    help='Base path containing experiment directories (default: ../data)')
parser.add_argument('--percentile', type=float, default=0.4, 
                    help='Percentile for cutoff calculation (default: 0.4 = 40th percentile)')
parser.add_argument('--min_files', type=int, default=3, 
                    help='Minimum number of required value files (default: 3)')
parser.add_argument('--skip_rows', type=int, default=0, 
                    help='Number of rows to skip when loading values (default: 0)')
args = parser.parse_args()

print(f"Base path:     {args.base_path}")
print(f"Percentile:    {args.percentile * 100:.0f}th")
print(f"Min files req: {args.min_files}")
print("=" * 60)

# Check if base path exists, otherwise generate synthetic data
if not os.path.exists(args.base_path):
    print(f"Warning: Base path not found: {args.base_path}")
    print("Generating synthetic experiment directories...")
    
    os.makedirs(args.base_path, exist_ok=True)
    
    # Create 3 synthetic experiments
    num_experiments = 3
    for exp_id in range(1, num_experiments + 1):
        exp_dir = os.path.join(args.base_path, f"exp_{exp_id}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Generate synthetic values with different distributions
        if exp_id == 1:
            # Low-skewed distribution
            train_vals = np.random.beta(2, 5, 100) * 0.5
            valid_vals = np.random.beta(2, 5, 30) * 0.5
            test_vals = np.random.beta(2, 5, 30) * 0.5
        elif exp_id == 2:
            # Mid-range distribution
            train_vals = np.random.beta(5, 5, 100) * 0.8
            valid_vals = np.random.beta(5, 5, 30) * 0.8
            test_vals = np.random.beta(5, 5, 30) * 0.8
        else:
            # High-skewed distribution
            train_vals = np.random.beta(5, 2, 100) * 0.9 + 0.1
            valid_vals = np.random.beta(5, 2, 30) * 0.9 + 0.1
            test_vals = np.random.beta(5, 2, 30) * 0.9 + 0.1
        
        # Save values
        np.savetxt(os.path.join(exp_dir, "train_vals.txt"), train_vals)
        np.savetxt(os.path.join(exp_dir, "valid_vals.txt"), valid_vals)
        np.savetxt(os.path.join(exp_dir, "test_vals.txt"), test_vals)
        
        print(f"✓ Created {exp_dir}")
    
    print()

# Find experiment directories
cutoffs = {}
experiment_count = 0

for item in sorted(os.listdir(args.base_path)):
    if item.startswith('.'):
        continue
    
    item_path = os.path.join(args.base_path, item)
    if not os.path.isdir(item_path):
        continue
    
    # Look for value files
    train_file = os.path.join(item_path, "train_vals.txt")
    valid_file = os.path.join(item_path, "valid_vals.txt")
    test_file = os.path.join(item_path, "test_vals.txt")
    
    # Count how many files exist
    existing_files = sum([
        os.path.exists(train_file),
        os.path.exists(valid_file),
        os.path.exists(test_file)
    ])
    
    if existing_files < args.min_files:
        continue
    
    try:
        # Load values from existing files
        values_list = []
        
        if os.path.exists(train_file):
            train_vals = np.loadtxt(train_file, skiprows=args.skip_rows)
            values_list.append(train_vals)
        
        if os.path.exists(valid_file):
            valid_vals = np.loadtxt(valid_file, skiprows=args.skip_rows)
            values_list.append(valid_vals)
        
        if os.path.exists(test_file):
            test_vals = np.loadtxt(test_file, skiprows=args.skip_rows)
            values_list.append(test_vals)
        
        if not values_list:
            continue
        
        # Merge all values and compute cutoff
        all_vals = np.hstack(values_list)
        sorted_vals = np.sort(all_vals)
        cutoff_idx = int(len(sorted_vals) * args.percentile)
        cutoff = sorted_vals[cutoff_idx]
        
        cutoffs[item] = cutoff
        experiment_count += 1
        
        print(f"'{item}': {cutoff:.6f},  [{len(all_vals)} total values]")
        
    except Exception as e:
        print(f"Warning: Could not process {item}: {e}")

print()
print("=" * 60)
print(f"Processed {experiment_count} experiment(s)")

if cutoffs:
    print()
    print("Copy this Python dict:")
    print("-" * 60)
    for exp_name in sorted(cutoffs.keys()):
        print(f"    '{exp_name}': {cutoffs[exp_name]:.6f},")
    print("-" * 60)
else:
    print("No experiments found with value files!")

print("Done")
