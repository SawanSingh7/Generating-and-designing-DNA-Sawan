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
parser = argparse.ArgumentParser(description="Multi-objective sequence optimization")
parser.add_argument('--log_dir', type=str, default=os.path.abspath("../logs"), help='Base log folder')
parser.add_argument('--log_name', type=str, default="pp_multi_test", help='Name for logging')
parser.add_argument('--generator', type=str, default=None, help="Location of generator model (.keras file)")
parser.add_argument('--predictor1', type=str, default=None, help="Location of first predictor model (.keras file)")
parser.add_argument('--predictor2', type=str, default=None, help="Location of second predictor model (.keras file)")
parser.add_argument('--target1', default="max", help="Target for predictor1: 'max', 'min', or float value")
parser.add_argument('--target2', default="min", help="Target for predictor2: 'max', 'min', or float value")
parser.add_argument('--target1_scale', default=1.0, type=float, help="Scaling for predictor1 cost (emphasize vs predictor2)")
parser.add_argument('--prior_weight', default=0., type=float, help="Weighting for latent prior term")
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

print("Multi-Objective Plug-and-Play Optimization (TensorFlow 2.x)")
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
predictor1 = None
predictor2 = None

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

# Try to load predictor1
if args.predictor1 and os.path.exists(args.predictor1):
    print(f"✓ Loading predictor1 from {args.predictor1}")
    predictor1 = tf.keras.models.load_model(args.predictor1)
else:
    if args.predictor1:
        print(f"Warning: Predictor1 file not found: {args.predictor1}")
    print("✓ Creating synthetic predictor1 model")
    predictor1 = create_simple_predictor(seq_len=36, vocab_size=len(charmap))
    pred1_path = os.path.join(logdir, "generated_predictor1.keras")
    predictor1.save(pred1_path)
    print(f"  Saved to {pred1_path}")

# Try to load predictor2
if args.predictor2 and os.path.exists(args.predictor2):
    print(f"✓ Loading predictor2 from {args.predictor2}")
    predictor2 = tf.keras.models.load_model(args.predictor2)
else:
    if args.predictor2:
        print(f"Warning: Predictor2 file not found: {args.predictor2}")
    print("✓ Creating synthetic predictor2 model")
    predictor2 = create_simple_predictor(seq_len=36, vocab_size=len(charmap))
    pred2_path = os.path.join(logdir, "generated_predictor2.keras")
    predictor2.save(pred2_path)
    print(f"  Saved to {pred2_path}")

print("-" * 60)

# Hyperparameters
batch_size = args.batch_size
alpha = args.prior_weight
step_size = args.step_size
latent_dim = args.latent_dim
target1_scale = args.target1_scale

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

print(f"Starting multi-objective optimization with {args.iterations} iterations")
print(f"Predictor1 target: {args.target1}, Predictor2 target: {args.target2}")
print(f"Optimizer: {args.optimizer}, Step size: {step_size}")
print(f"Batch size: {batch_size}, Latent dim: {latent_dim}")
print("-" * 60)

# Optimization loop
means1 = []
maxes1 = []
mins1 = []
means2 = []
maxes2 = []
mins2 = []

def parse_target(target_str):
    """Parse target string to determine optimization direction"""
    if target_str == "max":
        return "max"
    elif target_str == "min":
        return "min"
    else:
        try:
            return float(target_str)
        except ValueError:
            raise ValueError(f"Unknown target: {target_str}")

target1_type = parse_target(args.target1)
target2_type = parse_target(args.target2)

def compute_loss(preds, target_type, scale=1.0):
    """Compute loss based on target type"""
    if target_type == "max":
        return -scale * tf.reduce_mean(preds)  # Negative for maximization
    elif target_type == "min":
        return scale * tf.reduce_mean(preds)
    else:
        # Target is a specific value
        return scale * 0.5 * (tf.reduce_mean(preds) - target_type) ** 2

for iteration in range(args.iterations):
    with tf.GradientTape() as tape:
        # Generate sequences
        gen_sequences = generator(latent_codes, training=False)
        
        # Score sequences with both predictors
        predictions1 = predictor1(gen_sequences, training=False)
        predictions2 = predictor2(gen_sequences, training=False)
        predictions1 = tf.squeeze(predictions1)
        predictions2 = tf.squeeze(predictions2)
        
        # Compute combined loss
        loss1 = compute_loss(predictions1, target1_type, scale=target1_scale)
        loss2 = compute_loss(predictions2, target2_type)
        loss = loss1 + loss2
        
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
    preds1_eval = predictor1(gen_sequences_eval, training=False)
    preds2_eval = predictor2(gen_sequences_eval, training=False)
    preds1_eval = tf.squeeze(preds1_eval).numpy()
    preds2_eval = tf.squeeze(preds2_eval).numpy()
    
    means1.append(np.mean(preds1_eval))
    maxes1.append(np.max(preds1_eval))
    mins1.append(np.min(preds1_eval))
    means2.append(np.mean(preds2_eval))
    maxes2.append(np.max(preds2_eval))
    mins2.append(np.min(preds2_eval))
    
    # Log progress
    true_iteration = iteration + checkpoint_baseline + 1
    if true_iteration == checkpoint_baseline + 1 or true_iteration % args.log_interval == 0:
        print(f"Iter {true_iteration:>4d}: "
              f"pred1={np.mean(preds1_eval):.4f}±{np.std(preds1_eval):.4f}  "
              f"pred2={np.mean(preds2_eval):.4f}±{np.std(preds2_eval):.4f}")
        
        # Plot progress
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot predictor1
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Predictor 1 Scores")
        ax1.plot(range(len(means1)), means1, color='C2', label='Mean', linewidth=2)
        if args.plot_mode == "fill":
            ax1.fill_between(range(len(means1)), mins1, maxes1, color='C0', alpha=0.3, label='Min/max')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Predictor 1 (Target: {args.target1})")
        
        # Plot predictor2
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Predictor 2 Scores")
        ax2.plot(range(len(means2)), means2, color='C3', label='Mean', linewidth=2)
        if args.plot_mode == "fill":
            ax2.fill_between(range(len(means2)), mins2, maxes2, color='C1', alpha=0.3, label='Min/max')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"Predictor 2 (Target: {args.target2})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, "scores_multi_opt.png"), dpi=100)
        plt.close()
        
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
print(f"Multi-objective optimization complete!")
print(f"Predictor 1 - Mean: {means1[-1]:.6f}, Best: {maxes1[-1]:.6f}, Worst: {mins1[-1]:.6f}")
print(f"Predictor 2 - Mean: {means2[-1]:.6f}, Best: {maxes2[-1]:.6f}, Worst: {mins2[-1]:.6f}")
print(f"Results saved to: {logdir}")
print("Done")
