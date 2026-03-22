"""
TensorFlow 2.x compatible predictor/binding prediction model training
"""
import os
import argparse
import sys
import tensorflow as tf
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib


# ==================== Model Architecture ====================

class Predictor(tf.keras.Model):
    """CNN-based predictor for binding affinity prediction"""
    def __init__(self, vocab_size, filter_size, num_filters, num_layers, hidden_size, 
                 final_activation='linear', name='predictor'):
        super(Predictor, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.final_activation = final_activation
        
        # Convolutional layers
        self.conv_layers = []
        for i in range(num_layers):
            conv = tf.keras.layers.Conv1D(
                filters=num_filters,
                kernel_size=filter_size,
                padding='same',
                activation='relu',
                name=f'conv_{i}'
            )
            self.conv_layers.append(conv)
        
        # Global average pooling
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        
        # Dense layers
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu', name='hidden')
        self.output_layer = tf.keras.layers.Dense(1, activation=final_activation, name='output')
    
    def call(self, inputs, training=False):
        # Convert one-hot to channels
        x = inputs
        
        # Apply convolutions
        for conv in self.conv_layers:
            x = conv(x)
        
        # Global pooling
        x = self.pool(x)
        
        # Dense layers
        x = self.dense1(x)
        x = self.output_layer(x)
        
        return x


# ==================== Command Line Arguments ====================

parser = argparse.ArgumentParser()
parser.add_argument('--data_loc', type=str, default='../data', help='Data location')
parser.add_argument('--data_start', type=int, default=0, help='Line number to start when parsing data')
parser.add_argument('--log_dir', type=str, default=os.path.abspath("../logs"), help='Base log folder')
parser.add_argument('--log_name', type=str, default="predictor", help='Name to use when logging')
parser.add_argument('--checkpoint', type=str, default=None, help='Filename of checkpoint to load')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--checkpoint_iters', type=int, default=1, help='Number of epochs before saving checkpoint')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=36, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna_nt_only", help="Which vocabulary to use")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for one-hot encodings")
parser.add_argument('--filter_size', type=int, default=8, help='Convolutional filter size')
parser.add_argument('--num_filters', type=int, default=32, help='Number of convolutional filters')
parser.add_argument('--num_layers', type=int, default=3, help='Number of convolutional layers')
parser.add_argument('--hidden_size', type=int, default=64, help='Size of hidden dense layer')
parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer")
parser.add_argument('--final_activation', type=str, default='linear', help='Final layer activation')
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
logdir, checkpoint_baseline = lib.log(args, samples_dir=False)

print(f"Logdir: {logdir}")
print(f"Vocab size: {vocab_size}")


# ==================== Model Setup ====================

model = Predictor(
    vocab_size=vocab_size,
    filter_size=args.filter_size,
    num_filters=args.num_filters,
    num_layers=args.num_layers,
    hidden_size=args.hidden_size,
    final_activation=args.final_activation
)

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

# Checkpoint
checkpoint = tf.train.Checkpoint(
    model=model,
    optimizer=optimizer
)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, os.path.join(logdir, 'checkpoints'), max_to_keep=5
)

# Load checkpoint if provided
if args.checkpoint:
    checkpoint.restore(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")


# ==================== Training Loop ====================

def prepare_data(sequences, labels, batch_size=args.batch_size, max_len=args.max_seq_len):
    """Prepare data generator"""
    # Convert sequences to one-hot
    one_hot_seqs = []
    for seq in sequences:
        seq = seq[:max_len]
        indices = np.array([charmap.get(c, 0) for c in seq])
        one_hot = I[indices]
        one_hot_seqs.append(one_hot)
    
    one_hot_seqs = np.array(one_hot_seqs)
    labels = np.array(labels, dtype=np.float32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((one_hot_seqs, labels))
    dataset = dataset.shuffle(len(sequences))
    dataset = dataset.batch(batch_size)
    
    return dataset


@tf.function
def train_step(sequences, targets):
    """Single training step"""
    with tf.GradientTape() as tape:
        predictions = model(sequences, training=True)
        mse_loss = tf.reduce_mean(tf.square(predictions - tf.expand_dims(targets, 1)))
    
    gradients = tape.gradient(mse_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return mse_loss


@tf.function
def eval_step(sequences, targets):
    """Evaluation step (no training)"""
    predictions = model(sequences, training=False)
    mse_loss = tf.reduce_mean(tf.square(predictions - tf.expand_dims(targets, 1)))
    return mse_loss


# ==================== Generate Synthetic Data ====================

def generate_synthetic_data(num_samples=1000, seq_len=36):
    """Generate synthetic DNA sequences and binding values for demonstration"""
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], seq_len))
        sequences.append(seq)
        # Synthetic label: count of 'A' normalized
        label = seq.count('A') / seq_len
        labels.append(label)
    
    return sequences, labels


print("\nTraining Predictor (TensorFlow 2.x)")
print("=" * 50)

# Generate synthetic data for demo (in practice, load real data)
print("Generating synthetic training data...")
train_sequences, train_labels = generate_synthetic_data(num_samples=500)
val_sequences, val_labels = generate_synthetic_data(num_samples=100)

train_dataset = prepare_data(train_sequences, train_labels)
val_dataset = prepare_data(val_sequences, val_labels)

# Training loop
train_losses = []
val_losses = []
epoch_list = []

for epoch in range(args.num_epochs):
    epoch_loss = []
    
    # Training
    for sequences, targets in train_dataset:
        loss = train_step(sequences, targets)
        epoch_loss.append(loss)
    
    mean_train_loss = tf.reduce_mean(epoch_loss).numpy()
    train_losses.append(mean_train_loss)
    
    # Validation
    val_loss_list = []
    for sequences, targets in val_dataset:
        val_loss = eval_step(sequences, targets)
        val_loss_list.append(val_loss)
    
    mean_val_loss = tf.reduce_mean(val_loss_list).numpy()
    val_losses.append(mean_val_loss)
    epoch_list.append(epoch + 1)
    
    print(f"Epoch {epoch + 1}/{args.num_epochs}: "
          f"train_loss={mean_train_loss:.5f}, val_loss={mean_val_loss:.5f}")
    
    # Save checkpoint
    if (epoch + 1) % args.checkpoint_iters == 0:
        checkpoint_manager.save()
        print(f"  -> Checkpoint saved")

# Plot training curves
print("\nTraining complete!")
if len(train_losses) > 0:
    lib.plot(epoch_list, train_losses, logdir, 'train_loss',
             xlabel="Epoch", ylabel="MSE Loss")
    lib.plot(epoch_list, val_losses, logdir, 'val_loss',
             xlabel="Epoch", ylabel="MSE Loss")
    print(f"Plots saved to {logdir}")
