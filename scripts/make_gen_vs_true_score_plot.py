"""
Score distribution comparison plotter - creates histograms and scatter plots
Compares training, ground truth, and generated sequence score distributions
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

print("Score Distribution Comparison Plotter")
print("=" * 60)

parser = argparse.ArgumentParser(description="Create comparison plots of score distributions")
parser.add_argument('--gen_scores', type=str, default=os.path.abspath("../data/gen_scores.txt"), 
                    help='Location of scores for generated sequences (default: ../data/gen_scores.txt)')
parser.add_argument('--gt_scores', type=str, default=os.path.abspath("../data/gt_scores.txt"), 
                    help='Location of ground truth scores (default: ../data/gt_scores.txt)')
parser.add_argument('--train_scores', type=str, default=os.path.abspath("../data/train_scores.txt"), 
                    help='Location of training scores (default: ../data/train_scores.txt)')
parser.add_argument('--train_skiprows', type=int, default=0, 
                    help='Row number where data starts in training set (default: 0)')
parser.add_argument('--out_loc', type=str, default=os.path.abspath("../data"), 
                    help='Where to save the final plot (default: ../data)')
parser.add_argument('--out_name', type=str, default="make_gen_vs_true_score_plot", 
                    help="Filename for final figure (default: make_gen_vs_true_score_plot)")
parser.add_argument('--format', default="png", type=str, 
                    help="Format for saving plot (default: png)")
parser.add_argument('--name', type=str, default="comparison", 
                    help='Name of dataset to use for plot title (default: comparison)')
parser.add_argument('--lower', default=0., type=float, 
                    help='Lower limit of scores to plot (default: 0.0)')
parser.add_argument('--upper', default=1., type=float, 
                    help='Upper limit of scores to plot (default: 1.0)')
parser.add_argument('--train_label', default="True data", type=str, 
                    help="Label to use for training data in plot (default: True data)")
parser.add_argument('--gt_label', default='Synthetic "ground truth" data', type=str, 
                    help="Label to use for ground-truth data in plot")
parser.add_argument('--gen_label', default='"Ground truth" scores of generated samples', type=str, 
                    help="Label to use for generated scores in plot")
parser.add_argument('--xlabel', default="Scores", type=str, 
                    help="X-axis label for plot (default: Scores)")
parser.add_argument('--ylabel', default="Counts (normalized)", type=str, 
                    help="Y-axis label for plot (default: Counts (normalized))")
parser.add_argument('--plot_type', default="hist", type=str, 
                    help="Type of plot to make: hist or scatter (default: hist)")
args = parser.parse_args()

print(f"Generated scores: {args.gen_scores}")
print(f"Ground truth scores: {args.gt_scores}")
print(f"Training scores: {args.train_scores}")
print(f"Plot type: {args.plot_type}")
print(f"Output location: {args.out_loc}")
print("=" * 60)

# Check if score files exist, otherwise generate synthetic data
print()
score_files = {
    'training': args.train_scores,
    'ground truth': args.gt_scores,
    'generated': args.gen_scores
}

missing_files = {}
for name, filepath in score_files.items():
    if filepath and not os.path.exists(filepath):
        missing_files[name] = filepath

if missing_files:
    print(f"Warning: {len(missing_files)} score file(s) not found")
    print("Generating synthetic score data for demonstration...")
    
    # Create data directory
    data_dir = os.path.dirname(args.train_scores)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic scores
    np.random.seed(42)
    num_train = 200
    num_gt = 100
    num_gen = 100
    
    # Different distributions for each type
    train_scores_syn = np.random.beta(2, 5, num_train)  # Left-skewed
    gt_scores_syn = np.random.beta(5, 2, num_gt)  # Right-skewed
    gen_scores_syn = np.random.normal(0.6, 0.15, num_gen)  # Centered
    gen_scores_syn = np.clip(gen_scores_syn, 0, 1)
    
    if 'training' in missing_files:
        np.savetxt(args.train_scores, train_scores_syn)
        print(f"✓ Generated {len(train_scores_syn)} training scores")
    
    if 'ground truth' in missing_files:
        np.savetxt(args.gt_scores, gt_scores_syn)
        print(f"✓ Generated {len(gt_scores_syn)} ground truth scores")
    
    if 'generated' in missing_files:
        np.savetxt(args.gen_scores, gen_scores_syn)
        print(f"✓ Generated {len(gen_scores_syn)} generated scores")
    
    print()

# Load and plot
rng = np.linspace(args.lower, args.upper, 100)
plt.close()
ax = plt.gca()

train_scores = None
gt_scores = None
gen_scores = None

try:
    if os.path.exists(args.train_scores):
        train_scores = np.loadtxt(args.train_scores, skiprows=args.train_skiprows)
    
    if os.path.exists(args.gt_scores):
        gt_scores = np.loadtxt(args.gt_scores, skiprows=1)
    
    # For scatter plot, don't load generated scores (only compare train vs gt)
    if os.path.exists(args.gen_scores) and args.plot_type != "scatter":
        gen_scores = np.loadtxt(args.gen_scores, skiprows=1)
    
    if args.plot_type == "hist":
        if train_scores is not None:
            plt.hist(train_scores, bins=rng, density=True, color='C0', alpha=0.5, label=args.train_label)
        if gt_scores is not None:
            plt.hist(gt_scores, bins=rng, density=True, color='C1', alpha=0.5, label=args.gt_label)
        if gen_scores is not None:
            plt.hist(gen_scores, bins=rng, density=True, color='C2', alpha=0.5, label=args.gen_label)
            plt.legend()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    elif args.plot_type == "scatter":
        if train_scores is not None and gt_scores is not None:
            if len(train_scores) > len(gt_scores):
                print("Warning: omitting remainder of training scores where there is no corresponding ground truth score")
                train_scores = train_scores[:len(gt_scores)]
            plt.scatter(train_scores, gt_scores, alpha=0.7, s=0.3)
            ax.margins(0.0)
            ax.set_aspect('equal')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            pearson_corr, pearson_pval = pearsonr(train_scores, gt_scores)
            spearman_corr, spearman_pval = spearmanr(train_scores, gt_scores)
            print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_pval:.4e})")
            print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_pval:.4e})")
            plt.legend([args.train_label, args.gt_label])
        else:
            print("Error: Need both training and ground truth scores for scatter plot")
            sys.exit(1)

except Exception as e:
    print(f"Error loading or plotting scores: {e}")
    sys.exit(1)

plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
title = f"Scores of sequences ('{args.name}' data)" if args.name else "Score Distributions"
plt.title(title)

# Ensure output directory exists
os.makedirs(args.out_loc, exist_ok=True)

out_file = ".".join([args.out_name, args.format])
out_file = os.path.join(args.out_loc, out_file)
plt.savefig(out_file, pad_inches=0.0, transparent=False)
plt.close()

print()
print(f"✓ Plot saved to {out_file}")
print("Done")