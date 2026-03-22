import os
import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib

# Command line arguments
parser = argparse.ArgumentParser(description="Plug-and-Play sequence optimization")
parser.add_argument('--log_dir', type=str, default=os.path.abspath("../logs"), help='Base log folder')
parser.add_argument('--log_name', type=str, default="pp_test", help='Name to use when logging this script')
parser.add_argument('--generator', type=str, default=None, help="Location of generator model (.keras file)")
parser.add_argument('--predictor', type=str, default=None, help="Location of predictor model (.keras file)")
parser.add_argument('--target', default="max", help="Optimization target: 'max', 'min', or a target score (float)")
parser.add_argument('--prior_weight', default=0., type=float, help="Weighting for the latent prior term")
parser.add_argument('--optimizer', type=str, default="adam", help="Optimizer: 'adam' or 'sgd'")
parser.add_argument('--step_size', type=float, default=1e-1, help="Step-size for optimization")
parser.add_argument('--vocab', type=str, default="dna", help="Vocabulary: 'dna', 'rna', 'dna_nt_only', 'rna_nt_only'")
parser.add_argument('--vocab_order', type=str, default=None, help="Custom one-hot order")
parser.add_argument('--noise', type=float, default=1e-5, help="Gradient noise scale")
parser.add_argument('--iterations', type=int, default=100, help="Number of optimization iterations")
parser.add_argument('--log_interval', type=int, default=25, help="Progress report interval")
parser.add_argument('--save_samples', type=bool, default=True, help="Save samples during optimization")
parser.add_argument('--plot_mode', type=str, default="fill", help="Plot mode: 'fill' or 'line'")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size for optimization")
parser.add_argument('--latent_dim', type=int, default=100, help="Latent dimension for generator")
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

# Set random seeds
if args.seed is not None:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

# Get vocabulary
charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)

# Set up logging
logdir, checkpoint_baseline = lib.log(args, samples_dir=args.save_samples)

print("Plug-and-Play Optimization (TensorFlow 2.x)")
print("=" * 60)

def create_simple_generator(seq_len=36, vocab_size=4, latent_dim=100):
    """Create a simple generator model for DNA sequences"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(seq_len * vocab_size, activation='relu'),
        tf.keras.layers.Reshape((seq_len, vocab_size)),
        tf.keras.layers.Softmax(axis=-1)
    ])
    return model

def create_simple_predictor(seq_len=36, vocab_size=4):
    """Create a simple CNN predictor model for DNA binding"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same', input_shape=(seq_len, vocab_size)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Load or create models
generator = None
predictor = None

# Try to load generator
if args.generator and os.path.exists(args.generator):
    print(f"✓ Loading generator from {args.generator}")
    generator = tf.keras.models.load_model(args.generator)
else:
    if args.generator:
        print(f"Warning: Generator file not found: {args.generator}")
    print("✓ Creating synthetic generator model")
    generator = create_simple_generator(seq_len=36, vocab_size=len(charmap), latent_dim=args.latent_dim)
    gen_model_path = os.path.join(logdir, "generated_generator.keras")
    generator.save(gen_model_path)
    print(f"  Saved to {gen_model_path}")

# Try to load predictor
if args.predictor and os.path.exists(args.predictor):
    print(f"✓ Loading predictor from {args.predictor}")
    predictor = tf.keras.models.load_model(args.predictor)
else:
    if args.predictor:
        print(f"Warning: Predictor file not found: {args.predictor}")
    print("✓ Creating synthetic predictor model")
    predictor = create_simple_predictor(seq_len=36, vocab_size=len(charmap))
    pred_model_path = os.path.join(logdir, "generated_predictor.keras")
    predictor.save(pred_model_path)
    print(f"  Saved to {pred_model_path}")

print("-" * 60)

# Hyperparameters
batch_size = args.batch_size
alpha = args.prior_weight
step_size = args.step_size
latent_dim = args.latent_dim

# Initialize latent codes as trainable variables
latent_codes = tf.Variable(
    tf.random.normal([batch_size, latent_dim]),
    trainable=True,
    name='latent_codes'
)

# Create optimizer
if args.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=step_size)
elif args.optimizer == "sgd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=step_size)
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

print(f"Starting optimization with {args.iterations} iterations")
print(f"Target: {args.target}, Optimizer: {args.optimizer}, Step size: {step_size}")
print(f"Batch size: {batch_size}, Latent dim: {latent_dim}")
print("-" * 60)

# Optimization loop
means = []
maxes = []
mins = []
dist = []

for iteration in range(args.iterations):
    with tf.GradientTape() as tape:
        # Generate sequences
        gen_sequences = generator(latent_codes, training=False)
        
        # Score sequences
        predictions = predictor(gen_sequences, training=False)
        predictions = tf.squeeze(predictions)
        
        # Compute loss based on target
        if args.target == "max":
            loss = -tf.reduce_mean(predictions)  # Negative for maximization
        elif args.target == "min":
            loss = tf.reduce_mean(predictions)
        else:
            # Target is a specific score
            try:
                target_val = float(args.target)
                loss = 0.5 * (tf.reduce_mean(predictions) - target_val) ** 2
            except ValueError:
                raise ValueError(f"Unknown target: {args.target}")
        
        # Add prior term if specified
        if alpha > 0:
            prior_loss = alpha * tf.reduce_sum(latent_codes ** 2) / batch_size
            loss = loss + prior_loss
    
    # Compute and apply gradients
    grads = tape.gradient(loss, latent_codes)
    if grads is not None:
        # Add noise to gradients
        noise = tf.random.normal(latent_codes.shape, stddev=args.noise)
        grads = grads + noise
        optimizer.apply_gradients([(grads, latent_codes)])
    
    # Get predictions for logging
    gen_sequences_eval = generator(latent_codes, training=False)
    preds_eval = predictor(gen_sequences_eval, training=False)
    preds_eval = tf.squeeze(preds_eval).numpy()
    
    means.append(np.mean(preds_eval))
    maxes.append(np.max(preds_eval))
    mins.append(np.min(preds_eval))
    dist.append(preds_eval)
    
    # Log progress
    true_iteration = iteration + checkpoint_baseline + 1
    if true_iteration == checkpoint_baseline + 1 or true_iteration % args.log_interval == 0:
        print(f"Iter {true_iteration:>4d}: score={preds_eval[0]:.6f}; mean={np.mean(preds_eval):.6f}; "
              f"max={np.max(preds_eval):.6f}; std={np.std(preds_eval):.6f}")
        
        # Plot progress
        plt.clf()
        plt.xlabel("Iteration")
        plt.ylabel("Predicted Scores")
        plt.plot(range(len(means)), means, color='C2', label='Mean score', linewidth=2)
        if args.plot_mode == "fill":
            plt.fill_between(range(len(means)), mins, maxes, color='C0', alpha=0.3, label='Min/max range')
        else:
            plt.plot(range(len(maxes)), maxes, 'C0--', alpha=0.5, label='Max')
            plt.plot(range(len(mins)), mins, 'C1--', alpha=0.5, label='Min')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(logdir, "scores_opt.png"), dpi=100)
        
        # Save samples
        if args.save_samples:
            samples_dir = os.path.join(logdir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            with open(os.path.join(samples_dir, f"samples_{true_iteration:05d}.txt"), "w") as f:
                for row in gen_sequences_eval.numpy():
                    seq = "".join(rev_charmap[n] for n in np.argmax(row, -1))
                    f.write(seq + "\n")

# Final summary
print("\n" + "=" * 60)
print(f"Optimization complete!")
print(f"Final mean score:  {means[-1]:.6f}")
print(f"Best score:        {maxes[-1]:.6f} (iteration {np.argmax(means)+checkpoint_baseline+1})")
print(f"Score range:       [{mins[-1]:.6f}, {maxes[-1]:.6f}]")
print(f"Results saved to:  {logdir}")
print("Done")

print("Done")