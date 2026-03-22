import tensorflow as tf
import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib

parser = argparse.ArgumentParser()
parser.add_argument('--latent', action='store_true', help="Include a single latent layer with softmax.")
parser.add_argument('--save_dir', type=str, default=os.path.abspath("../logs/baselines"), help='Base save folder')
parser.add_argument('--save_name', type=str, default="baseline_generator", help='Name to use when saving this model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=36, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
args = parser.parse_args()

# Fix vocabulary of model
charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)

# Build baseline generator model
class BaselineGenerator(tf.keras.Model):
    """Baseline generator that produces one-hot or soft-encoded DNA sequences"""
    def __init__(self, batch_size, max_seq_len, vocab_size, use_softmax=False):
        super(BaselineGenerator, self).__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.use_softmax = use_softmax
        
        # Latent variables
        self.latent_vars = self.add_weight(
            shape=(batch_size, max_seq_len * vocab_size),
            initializer=tf.random_normal_initializer(),
            trainable=True,
            name='latent_vars'
        )
    
    def call(self, inputs=None, training=False):
        # Reshape latent variables to sequence format
        reshaped = tf.reshape(self.latent_vars, [self.batch_size, self.max_seq_len, self.vocab_size])
        
        # Apply softmax if requested, otherwise use raw latents
        if self.use_softmax:
            output = tf.nn.softmax(reshaped, axis=-1)
        else:
            output = reshaped
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'batch_size': self.batch_size,
            'max_seq_len': self.max_seq_len,
            'vocab_size': self.vocab_size,
            'use_softmax': self.use_softmax,
        })
        return config

# Create model
model = BaselineGenerator(
    batch_size=args.batch_size,
    max_seq_len=args.max_seq_len,
    vocab_size=vocab_size,
    use_softmax=args.latent
)

# Build model by calling it once
dummy_input = tf.zeros((args.batch_size, args.max_seq_len, vocab_size))
_ = model(dummy_input)

# Save model
save_loc = os.path.join(os.path.expanduser(args.save_dir), args.save_name)
os.makedirs(save_loc, exist_ok=True)

if args.latent:
    name = "latent_generator"
else:
    name = "no_latent_generator"

save_path = os.path.join(save_loc, name + ".keras")
model.save(save_path)

print(f"✓ Model saved to {save_path}")
print("Done")