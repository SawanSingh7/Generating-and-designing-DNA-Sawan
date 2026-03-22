import numpy as np
import tensorflow as tf
import os
import sys
import argparse
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib
from lib.explicit import max_match

# (Optional) command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default=os.path.abspath("../logs/baselines"), help='Folder to save the model in')
parser.add_argument('--save_name', type=str, default="max_match_predictor", help='Name to use when saving this model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--max_seq_len', type=int, default=50, help="Maximum sequence length of data")
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--vocab_order', type=str, default=None, help="Specific order for the one-hot encodings of vocab characters")
parser.add_argument('--pwm_file', type=str, default=None, help='CSV file to load PWM pattern from')
parser.add_argument('--pattern', type=str, default=None, help='Hard-pattern string to match')
parser.add_argument('--padding', type=str, default="VALID", help='Padding to use in conv1d. Options are "SAME" and "VALID"')
parser.add_argument('--stride', type=int, default=1, help='Stride size for conv1d.')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

charmap, rev_charmap = lib.dna.get_vocab(args.vocab, args.vocab_order)
vocab_size = len(charmap)
I = np.eye(vocab_size)

if args.pwm_file and args.pattern:
    raise Exception("Please specify only one of pwm_file or pattern")
elif not (args.pwm_file or args.pattern):
    raise Exception("Please specify either a pwm_file or a pattern")
elif args.pwm_file:
    pwm = np.loadtxt(args.pwm_file)
elif args.pattern:
    indices = [charmap[c] for c in args.pattern]
    pwm = np.vstack([I[idx] for idx in indices])

# Build TensorFlow 2.x model
class MaxMatchPredictor(tf.keras.Model):
    """Pattern matching predictor using max match"""
    def __init__(self, pwm, padding='VALID', stride=1):
        super(MaxMatchPredictor, self).__init__()
        self.pwm = tf.constant(pwm, dtype=tf.float32)
        self.padding = padding
        self.stride = stride
    
    def call(self, inputs, training=False):
        # inputs shape: (batch, seq_len, vocab_size)
        return max_match(inputs, self.pwm, padding=self.padding, stride=self.stride)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'pwm': self.pwm.numpy().tolist(),
            'padding': self.padding,
            'stride': self.stride,
        })
        return config

# Create and save model
model = MaxMatchPredictor(pwm, padding=args.padding, stride=args.stride)

# Build model
dummy_input = tf.zeros((args.batch_size, args.max_seq_len, vocab_size), dtype=tf.float32)
_ = model(dummy_input)

save_loc = os.path.join(os.path.expanduser(args.save_dir), args.save_name)
os.makedirs(save_loc, exist_ok=True)

save_path = os.path.join(save_loc, "predictor.keras")
model.save(save_path)

# Save pattern information
if args.pattern:
    with open(os.path.join(save_loc, "pattern.txt"), "w") as f:
        f.write(args.pattern)
elif args.pwm_file:
    shutil.copy(args.pwm_file, os.path.join(save_loc, "pattern.txt"))

print(f"✓ Model saved to {save_path}")
print("Done")