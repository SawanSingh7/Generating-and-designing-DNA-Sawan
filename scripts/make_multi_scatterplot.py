"""
Multi-target scatter plot generator - compares scores across two targets
Creates scatter plots comparing training and generated sequence scores for dual objectives
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.rcParams.update({'font.size': 14})

print("Multi-Target Scatter Plot Generator")
print("=" * 60)

parser = argparse.ArgumentParser(description="Create multi-target comparison scatter plots")
parser.add_argument('--gen_scores1', type=str, default=os.path.abspath("../data/gen_scores1.txt"), 
                    help='Location of target 1 scores for generated sequences (default: ../data/gen_scores1.txt)')
parser.add_argument('--gen_scores2', type=str, default=os.path.abspath("../data/gen_scores2.txt"), 
                    help='Location of target 2 scores for generated sequences (default: ../data/gen_scores2.txt)')
parser.add_argument('--train_scores1', type=str, default=os.path.abspath("../data/train_scores1.txt"), 
                    help='Location of training scores for target 1 (default: ../data/train_scores1.txt)')
parser.add_argument('--train_scores2', type=str, default=os.path.abspath("../data/train_scores2.txt"), 
                    help='Location of training scores for target 2 (default: ../data/train_scores2.txt)')
parser.add_argument('--train_skiprows1', type=int, default=0, 
                    help='Row number where data starts in target 1 training set (default: 0)')
parser.add_argument('--train_skiprows2', type=int, default=0, 
                    help='Row number where data starts in target 2 training set (default: 0)')
parser.add_argument('--out_loc', type=str, default=os.path.abspath("../data"), 
                    help='Where to save the final plot (default: ../data)')
parser.add_argument('--out_name', type=str, default="make_multi_scatterplot", 
                    help="Filename for final figure (default: make_multi_scatterplot)")
parser.add_argument('--format', default="png", type=str, 
                    help="Format for saving plot (default: png)")
parser.add_argument('--lower', default=0., type=float, 
                    help='Lower limit of scores to plot (default: 0.0)')
parser.add_argument('--upper', default=1., type=float, 
                    help='Upper limit of scores to plot (default: 1.0)')
parser.add_argument('--train_label', default="Training data", type=str, 
                    help="Label to use for training data in plot (default: Training data)")
parser.add_argument('--gen_label', default='Generated data', type=str, 
                    help="Label to use for generated scores in plot (default: Generated data)")
parser.add_argument('--xlabel', default="Target 1 scores", type=str, 
                    help="X-axis label for plot (default: Target 1 scores)")
parser.add_argument('--ylabel', default="Target 2 scores", type=str, 
                    help="Y-axis label for plot (default: Target 2 scores)")
args = parser.parse_args()

print(f"Train target 1: {args.train_scores1}")
print(f"Train target 2: {args.train_scores2}")
print(f"Gen target 1:   {args.gen_scores1}")
print(f"Gen target 2:   {args.gen_scores2}")
print(f"Output:         {args.out_loc}")
print("=" * 60)

# Check if score files exist, otherwise generate synthetic data
print()
score_files = {
    'train_scores1': args.train_scores1,
    'train_scores2': args.train_scores2,
    'gen_scores1': args.gen_scores1,
    'gen_scores2': args.gen_scores2
}

missing_files = {k: v for k, v in score_files.items() if not os.path.exists(v)}

if missing_files:
    print(f"Warning: {len(missing_files)} score file(s) not found")
    print("Generating synthetic score data for demonstration...")
    
    # Create data directory
    data_dir = os.path.dirname(args.train_scores1)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic scores
    np.random.seed(42)
    num_train = 150
    num_gen = 80
    
    # Training scores: two correlated but slightly different distributions
    train_scores_1 = np.random.beta(2, 5, num_train)  # Left-skewed
    train_scores_2 = train_scores_1 + np.random.normal(0, 0.1, num_train)  # Correlated
    train_scores_2 = np.clip(train_scores_2, 0, 1)
    
    # Generated scores: similar pattern but with different mean
    gen_scores_1 = np.random.beta(5, 2, num_gen)  # Right-skewed
    gen_scores_2 = gen_scores_1 + np.random.normal(0, 0.1, num_gen)  # Correlated
    gen_scores_2 = np.clip(gen_scores_2, 0, 1)
    
    if 'train_scores1' in missing_files:
        np.savetxt(args.train_scores1, train_scores_1)
        print(f"✓ Generated {len(train_scores_1)} training target 1 scores")
    
    if 'train_scores2' in missing_files:
        np.savetxt(args.train_scores2, train_scores_2)
        print(f"✓ Generated {len(train_scores_2)} training target 2 scores")
    
    if 'gen_scores1' in missing_files:
        np.savetxt(args.gen_scores1, gen_scores_1)
        print(f"✓ Generated {len(gen_scores_1)} generated target 1 scores")
    
    if 'gen_scores2' in missing_files:
        np.savetxt(args.gen_scores2, gen_scores_2)
        print(f"✓ Generated {len(gen_scores_2)} generated target 2 scores")
    
    print()

# Load data
try:
    train_scores_1 = np.loadtxt(args.train_scores1, skiprows=args.train_skiprows1)
    train_scores_2 = np.loadtxt(args.train_scores2, skiprows=args.train_skiprows2)
    gen_scores_1 = np.loadtxt(args.gen_scores1, skiprows=args.train_skiprows1)
    gen_scores_2 = np.loadtxt(args.gen_scores2, skiprows=args.train_skiprows2)
    
    print(f"Loaded scores:")
    print(f"  Train: {len(train_scores_1)} points")
    print(f"  Gen:   {len(gen_scores_1)} points")
    print()
except Exception as e:
    print(f"Error loading scores: {e}")
    sys.exit(1)
# Create plot
plt.close()
ax = plt.gca()

plt.scatter(train_scores_1, train_scores_2, alpha=0.5, marker='o', s=50, label=args.train_label)
plt.scatter(gen_scores_1, gen_scores_2, alpha=0.5, marker='^', s=50, label=args.gen_label)

ax.margins(0.0)
ax.set_aspect('equal')
plt.xlim([args.lower, args.upper])
plt.ylim([args.lower, args.upper])

rng = np.linspace(args.lower, args.upper, 6)
ax.set_xticks(rng)
ax.set_yticks(rng)

plt.legend()

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)

# Ensure output directory exists
os.makedirs(args.out_loc, exist_ok=True)

out_file = ".".join([args.out_name, args.format])
out_file = os.path.join(args.out_loc, out_file)

plt.savefig(out_file, pad_inches=0.0, transparent=False)
plt.close()

print(f"✓ Plot saved to {out_file}")
print("Done")