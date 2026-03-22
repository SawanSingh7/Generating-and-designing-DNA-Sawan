"""
Interpolation plot generator - visualizes latent space interpolations
Creates plots showing how model outputs vary across latent space dimensions
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import argparse
import pickle
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Interpolation Plot Generator")
print("=" * 60)

parser = argparse.ArgumentParser(description="Create interpolation plots from latent space data")
parser.add_argument('--data_file', type=str, default=os.path.abspath("../data/interpolation_data.pkl"), 
                    help='File with pickled interpolation data (default: ../data/interpolation_data.pkl)')
parser.add_argument('--num_plots', type=int, default=7, 
                    help='Number of subplots to show (default: 7)')
parser.add_argument('--out_file', type=str, default="make_interpolation_plot.svg", 
                    help='Filename to save created plot (default: make_interpolation_plot.svg)')
args = parser.parse_args()

print(f"Data file: {args.data_file}")
print(f"Num plots: {args.num_plots}")
print(f"Output file: {args.out_file}")
print("=" * 60)

# Check if data file exists, otherwise generate synthetic data
if not os.path.exists(args.data_file):
    print(f"Warning: Data file not found: {args.data_file}")
    print("Generating synthetic interpolation data for demonstration...")
    
    # Create data directory
    data_dir = os.path.dirname(args.data_file)
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate synthetic interpolation data
    # Shape: (latent_steps, num_sequences, vocab_size)
    latent_steps = 20
    num_seqs = args.num_plots
    vocab_size = 4  # DNA bases
    
    # Create smooth interpolation curves for each sequence
    data = np.zeros((latent_steps, num_seqs, vocab_size))
    for seq_idx in range(num_seqs):
        for char_idx in range(vocab_size):
            # Create smooth sine-wave-like interpolation
            t = np.linspace(0, 2*np.pi, latent_steps)
            data[:, seq_idx, char_idx] = 0.5 + 0.3 * np.sin(t + seq_idx + char_idx)
            data[:, seq_idx, char_idx] = np.clip(data[:, seq_idx, char_idx], 0, 1)
    
    # Save synthetic data
    with open(args.data_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Generated synthetic interpolation data")
    print(f"  Shape: {data.shape} (latent_steps, num_sequences, vocab_size)")
    print(f"✓ Saved to {args.data_file}")
    print()

# Load data
try:
    with open(args.data_file, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded interpolation data: shape {data.shape}")
    print()
except Exception as e:
    print(f"Error loading data file: {e}")
    sys.exit(1)

# Define colors for DNA bases
colours = np.array([(0, 213, 0),    # green (A)
                    (0, 0, 192),    # blue (C)
                    (255, 170, 0),  # yellow (G)
                    (213, 0, 0)],   # red (T)
                   dtype=float) / 255

# Adjust number of plots to match data if needed
num_plots = min(args.num_plots, data.shape[1])
if num_plots != args.num_plots:
    print(f"Note: Adjusting num_plots from {args.num_plots} to {num_plots} (data has {data.shape[1]} sequences)")

# Create subplots
f, axes = plt.subplots(1, num_plots, sharex=True, sharey=True, figsize=(num_plots*2, 4))

# Handle single plot case
if num_plots == 1:
    axes = [axes]

# Plot interpolation curves for each sequence and base
for seq_idx, axis in enumerate(axes):
    for char_idx in range(min(4, data.shape[2])):  # Min with vocab size
        plt_data = data[:, seq_idx, char_idx]  # data has indices (latent_idx, seq_idx, vocab_idx)
        axis.plot(plt_data[::-1], range(len(plt_data)), color=colours[char_idx], linewidth=1.5)

# Fine-tune figure; make subplots close to each other
f.subplots_adjust(hspace=0, wspace=0.1)
plt.setp([a.get_yticklabels() for a in f.axes], visible=False)
plt.setp([a.get_yaxis() for a in f.axes], visible=False)
plt.setp([a.get_xticklabels() for a in f.axes], visible=False)

# Determine output format from filename
if args.out_file.endswith('.pdf'):
    fmt = 'pdf'
elif args.out_file.endswith('.png'):
    fmt = 'png'
else:
    fmt = 'svg'

plt.savefig(args.out_file, format=fmt, bbox_inches='tight')
plt.close()

print(f"✓ Plot saved to {args.out_file}")
print("Done")