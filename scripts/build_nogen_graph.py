import numpy as np
import tensorflow as tf
import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default=os.path.abspath("../logs/baselines"), help='Folder to save the model in')
parser.add_argument('--save_name', type=str, default="nogen_generator", help='Name to use when saving this model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
args = parser.parse_args()

charmap, rev_charmap = lib.dna.get_vocab(args.vocab)
vocab_size = len(charmap)

# Build TensorFlow 2.x model  
class NoGenGenerator(tf.keras.Model):
    """Generator-free baseline with latent variables"""
    def __init__(self, batch_size, max_seq_len, vocab_size):
        super(NoGenGenerator, self).__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Latent variables
        self.latents = self.add_weight(
            shape=(batch_size, max_seq_len * vocab_size),
            initializer=tf.constant_initializer(0.),
            trainable=True,
            name='latent_vars'
        )
    
    def call(self, inputs=None, training=False):
        # Reshape latents to sequence format
        output = tf.reshape(self.latents, [self.batch_size, self.max_seq_len, self.vocab_size])
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'batch_size': self.batch_size,
            'max_seq_len': self.max_seq_len,
            'vocab_size': self.vocab_size,
        })
        return config

# Create and save model
model = NoGenGenerator(args.batch_size, args.max_seq_len, vocab_size)

# Build model
dummy_input = tf.zeros((args.batch_size, args.max_seq_len, vocab_size), dtype=tf.float32)
_ = model(dummy_input)

save_loc = os.path.join(os.path.expanduser(args.save_dir), args.save_name)
os.makedirs(save_loc, exist_ok=True)

save_path = os.path.join(save_loc, "generator.keras")
model.save(save_path)

print(f"✓ Model saved to {save_path}")
print("Done")