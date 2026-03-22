"""
TensorFlow 2.x compatible GAN training script
"""
import os
import argparse
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib

# ==================== TensorFlow 2.x Model Classes ====================

class Generator(tf.keras.Model):
    """Generator model for DNA sequence generation"""
    def __init__(self, latent_dim, output_size, num_layers, hidden_dim, name='generator'):
        super(Generator, self).__init__(name=name)
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.dense_layers = []
        for i in range(num_layers):
            if i == 0:
                layer = tf.keras.layers.Dense(hidden_dim, activation='relu')
            elif i == num_layers - 1:
                layer = tf.keras.layers.Dense(output_size, activation='sigmoid')
            else:
                layer = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.dense_layers.append(layer)
    
    def call(self, latents, training=False):
        x = latents
        for layer in self.dense_layers:
            x = layer(x)
        return x


class Discriminator(tf.keras.Model):
    """Discriminator model for DNA sequence discrimination"""
    def __init__(self, input_size, num_layers, hidden_dim, name='discriminator'):
        super(Discriminator, self).__init__(name=name)
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.dense_layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                layer = tf.keras.layers.Dense(1)
            else:
                layer = tf.keras.layers.Dense(hidden_dim, activation='relu')
            self.dense_layers.append(layer)
    
    def call(self, inputs, training=False):
        # Flatten if needed
        if len(inputs.shape) > 2:
            x = tf.reshape(inputs, [inputs.shape[0], -1])
        else:
            x = inputs
        
        for layer in self.dense_layers:
            x = layer(x)
        return x


# ==================== Command line arguments ====================

checkpoint = None

parser = argparse.ArgumentParser()
parser.add_argument('--generic', action='store_true', help="Generate generic data on the fly (ignores data_loc and data_start args)")
parser.add_argument('--data_loc', type=str, default=os.path.abspath("../data"), help='Data location (default: ../data)')
parser.add_argument('--data_start', type=int, default=0, help='Line number to start when parsing data (useful for ignoring header)')
parser.add_argument('--log_dir', type=str, default=os.path.abspath("../logs"), help='Base log folder')
parser.add_argument('--log_name', type=str, default="gan_test", help='Name to use when logging this model')
parser.add_argument('--checkpoint', type=str, default=checkpoint, help='Filename of checkpoint to load')
parser.add_argument('--model_type', type=str, default="mlp", help='Which type of model architecture to use')
parser.add_argument('--train_iters', type=int, default=100, help='Number of iterations to train GAN for')
parser.add_argument('--disc_iters', type=int, default=5, help='Number of iterations to train discriminator for at each training step')
parser.add_argument('--checkpoint_iters', type=int, default=250, help='Number of iterations before saving checkpoint')
parser.add_argument('--latent_dim', type=int, default=100, help='Size of latent space')
parser.add_argument('--gen_dim', type=int, default=100, help='Generator dimension parameter')
parser.add_argument('--disc_dim', type=int, default=100, help='Discriminator dimension parameter')
parser.add_argument('--gen_layers', type=int, default=5, help='How many layers for generator')
parser.add_argument('--disc_layers', type=int, default=5, help='How many layers for discriminator')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=36, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings")
parser.add_argument('--annotate', action='store_true', help="Include annotation as part of training/generation process?")
parser.add_argument('--validate', action='store_false', help="Whether to use validation set")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer")
parser.add_argument('--lmbda', type=float, default=10., help='Lipschitz penalty hyperparameter')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
args = parser.parse_args()

# Set seeds
if args.seed is not None:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

# Fix vocabulary
charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)
I = np.eye(vocab_size)

# Organize model logs/checkpoints
logdir, checkpoint_baseline = lib.log(args, samples_dir=True)

# Compute data dimensions
if args.annotate:
    data_enc_dim = vocab_size + 1
else:
    data_enc_dim = vocab_size
data_size = args.max_seq_len * data_enc_dim

# Build models
generator = Generator(args.latent_dim, data_size, args.gen_layers, args.gen_dim)
discriminator = Discriminator(data_size, args.disc_layers, args.disc_dim)

# Create optimizers
gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5, beta_2=0.9)
disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.5, beta_2=0.9)

# Create checkpoint
checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer
)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, os.path.join(logdir, 'checkpoints'), max_to_keep=5
)


# ==================== Training functions ====================

@tf.function
def discriminator_loss(real_data, gen_data, real_scores, gen_scores, interp_data, interp_scores, lmbda):
    """Compute discriminator loss with gradient penalty"""
    disc_diff = tf.reduce_mean(gen_scores) - tf.reduce_mean(real_scores)
    
    # Gradient penalty
    with tf.GradientTape() as tape:
        tape.watch(interp_data)
        interp_score = discriminator(interp_data, training=True)
    grads = tape.gradient(interp_score, interp_data)
    
    # Compute gradient norm
    if len(grads.shape) > 2:
        grad_norms = tf.norm(grads, axis=list(range(1, len(grads.shape))))
    else:
        grad_norms = tf.norm(grads, axis=1)
    
    grad_penalty = lmbda * tf.reduce_mean((grad_norms - 1.0) ** 2)
    
    return disc_diff + grad_penalty


@tf.function
def generator_loss(gen_scores):
    """Compute generator loss"""
    return -tf.reduce_mean(gen_scores)


def train_step(real_data, lmbda):
    """Single training step"""
    batch_size = tf.shape(real_data)[0]
    noise = tf.random.normal([batch_size, args.latent_dim])
    
    # Create interpolated data
    eps = tf.random.uniform([batch_size, 1] + [1] * (len(real_data.shape) - 2))
    gen_data = generator(noise, training=True)
    gen_data_reshaped = tf.reshape(gen_data, real_data.shape)
    interp_data = eps * real_data + (1.0 - eps) * gen_data_reshaped
    
    # Train discriminator
    disc_losses = []
    for _ in range(args.disc_iters):
        noise = tf.random.normal([batch_size, args.latent_dim])
        gen_data = generator(noise, training=True)
        gen_data_reshaped = tf.reshape(gen_data, real_data.shape)
        
        with tf.GradientTape() as tape:
            real_scores = discriminator(real_data, training=True)
            gen_scores = discriminator(gen_data_reshaped, training=True)
            interp_scores = discriminator(interp_data, training=True)
            disc_loss = discriminator_loss(real_data, gen_data_reshaped, real_scores, 
                                          gen_scores, interp_data, interp_scores, lmbda)
        
        disc_grads = tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        disc_losses.append(disc_loss)
    
    # Train generator
    noise = tf.random.normal([batch_size, args.latent_dim])
    with tf.GradientTape() as tape:
        gen_data = generator(noise, training=True)
        gen_data_reshaped = tf.reshape(gen_data, real_data.shape)
        gen_scores = discriminator(gen_data_reshaped, training=True)
        gen_loss = generator_loss(gen_scores)
    
    gen_grads = tape.gradient(gen_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    
    return tf.reduce_mean(disc_losses), gen_loss


# ==================== Data loading ====================

if args.generic:
    if args.annotate:
        raise Exception("args `annotate` and `generic` are incompatible.")

    def data_generator(batch_size=args.batch_size, seq_len=args.max_seq_len):
        while True:
            samples = np.random.choice(vocab_size, [batch_size, seq_len])
            data = np.vstack([np.expand_dims(I[vec], 0) for vec in samples])
            yield data.astype(np.float32)
    
    train_data = data_generator()
else:
    data = lib.dna.load(args.data_loc, vocab_order=args.vocab_order, 
                       max_seq_len=args.max_seq_len,
                       data_start_line=args.data_start, vocab=args.vocab, 
                       valid=args.validate, annotate=args.annotate)
    
    if args.validate:
        split = len(data) // 2
        train_data_raw = data[:split]
        valid_data_raw = data[split:]
    else:
        train_data_raw = data
    
    if args.annotate:
        train_data_raw = np.concatenate(train_data_raw, 2)
        if args.validate:
            valid_data_raw = np.concatenate(valid_data_raw, 2)
    elif isinstance(train_data_raw, list):
        train_data_raw = np.concatenate(train_data_raw, 0)
        if args.validate:
            valid_data_raw = np.concatenate(valid_data_raw, 0)
    
    train_data_raw = train_data_raw.astype(np.float32)
    
    def create_generator(data, batch_size=args.batch_size):
        num_batches = len(data) // batch_size
        while True:
            for i in range(num_batches):
                batch = data[i * batch_size:(i + 1) * batch_size]
                yield batch
    
    train_data = create_generator(train_data_raw)


# ==================== Training loop ====================

print("Training GAN (TensorFlow 2.x)")
print("=" * 50)

fixed_latents = tf.constant(np.random.normal(size=[args.batch_size, args.latent_dim]).astype(np.float32))
train_disc_costs = []
train_counts = []
valid_costs = []
valid_counts_list = []

for iteration in range(args.train_iters):
    true_count = iteration + 1 + checkpoint_baseline
    train_counts.append(true_count)
    
    # Get batch
    real_batch = next(train_data)
    real_batch = tf.constant(real_batch)
    
    # Train step
    disc_loss, gen_loss = train_step(real_batch, args.lmbda)
    train_disc_costs.append(disc_loss.numpy())
    
    # Checkpoint
    if true_count % args.checkpoint_iters == 0:
        # Generate samples
        gen_samples = generator(fixed_latents, training=False).numpy()
        gen_samples = gen_samples.reshape([args.batch_size, args.max_seq_len, data_enc_dim])
        
        print(f"Iteration {true_count}: disc_loss={disc_loss:.5f}, gen_loss={gen_loss:.5f}")
        
        # Save samples
        lib.save_samples(logdir, gen_samples, true_count, rev_charmap, annotated=args.annotate)
        
        # Save checkpoint
        checkpoint_manager.save()
        
        # Plot
        name = "train_disc_cost"
        if checkpoint_baseline > 0:
            name += f"_{checkpoint_baseline}"
        lib.plot(train_counts, train_disc_costs, logdir, name, 
                xlabel="Iteration", ylabel="Discriminator cost")

print("Done")
