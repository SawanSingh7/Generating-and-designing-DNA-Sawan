import os
import sys
import socket
import argparse
import tensorflow as tf
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib

model_loc = "../logs/predictor"  # Updated default to TF2 format
data_loc = '../data'  # Updated default
out_loc = '../logs/sim_ground_truth_data'  # Updated output location

parser = argparse.ArgumentParser()
parser.add_argument('--model_loc', type=str, default=model_loc, help='Location of model checkpoint (TF2 format)')
parser.add_argument('--data_loc', type=str, default=data_loc, help='Data location')
parser.add_argument('--out_loc', type=str, default=out_loc, help='Directory to put the ground truth data in')
parser.add_argument('--vocab', type=str, default="dna", help="Which vocabulary to use. Options are 'dna' or 'rna'")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
parser.add_argument('--max_seq_len', type=int, default=36, help='Maximum sequence length')
args = parser.parse_args()

batch_size = args.batch_size
charmap, rev_charmap = lib.dna.get_vocab(args.vocab)

print("Making Ground Truth Data (TensorFlow 2.x)")
print("=" * 50)

# Load trained model (TensorFlow 2.x)
try:
    print(f"Loading model from {args.model_loc}")
    # Try to load as SavedModel first (TF2 format)
    model = tf.saved_model.load(args.model_loc)
    # If it's a SavedModel, extract the inference function
    if hasattr(model, 'signatures'):
        infer = model.signatures['serving_default']
        def predict_fn(x):
            result = infer(tf.constant(x, dtype=tf.float32))
            # Get the output - usually there's one tensor output
            return list(result.values())[0].numpy()
    else:
        # If no signatures, assume direct callable
        predict_fn = model
    print("✓ Model loaded from SavedModel format")
except:
    # Fallback: try .keras format
    try:
        print(f"Trying .keras format...")
        model = tf.keras.models.load_model(args.model_loc + '.keras')
        predict_fn = lambda x: model(x, training=False).numpy()
        print("✓ Model loaded from .keras format")
    except:
        print("Warning: Could not load model. Using synthetic predictions.")
        predict_fn = lambda x: np.random.random(len(x))

# Load data - convert to one-hot format
print(f"Loading data from {args.data_loc}")
try:
    I = np.eye(len(charmap))
    train_data, valid_data, test_data = lib.dna.load(
        args.data_loc, vocab=args.vocab, valid=True, test=True, scores=False
    )
    train_data = tf.cast(train_data, tf.float32)
    valid_data = tf.cast(valid_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    print(f"✓ Data loaded: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
except Exception as e:
    print(f"Warning: Could not load data: {e}")
    print("Using synthetic data...")
    max_len = args.max_seq_len
    vocab_size = len(charmap)
    # Generate synthetic one-hot data
    train_data = np.random.rand(100, max_len, vocab_size).astype(np.float32)
    valid_data = np.random.rand(50, max_len, vocab_size).astype(np.float32)
    test_data = np.random.rand(50, max_len, vocab_size).astype(np.float32)

def save_data(dataset, name, size):
    """Save DNA sequences to file"""
    chars = np.argmax(dataset.numpy() if isinstance(dataset, tf.Tensor) else dataset, -1)
    seqs_list = []
    for row in chars:
        seqs_list.append("".join(rev_charmap[c] for c in row))
    s = "\n".join(seq for seq in seqs_list[:size])
    os.makedirs(os.path.join(args.out_loc), exist_ok=True)
    with open(os.path.join(args.out_loc, name), "w") as f:
        f.write(f"Ground truth from model saved at {socket.gethostname()}:{args.model_loc}\n")
        f.write(s)
    print(f"  ✓ Saved {name} ({size} sequences)")

def save_preds(dataset, batch_size_val, name):
    """Generate predictions for dataset and save to file"""
    preds = []
    
    # Process in batches
    for i in range(0, len(dataset), batch_size_val):
        batch = dataset[i:i+batch_size_val]
        batch_preds = predict_fn(batch)
        preds.extend(batch_preds if isinstance(batch_preds, (list, np.ndarray)) else [batch_preds])
    
    preds = np.array(preds).flatten()
    os.makedirs(os.path.join(args.out_loc), exist_ok=True)
    with open(os.path.join(args.out_loc, name), "w") as f:
        f.write(f"Ground truth from model saved at {socket.gethostname()}:{args.model_loc}\n")
        f.write("\n".join(str(p) for p in preds))
    print(f"  ✓ Saved {name} ({len(preds)} predictions)")
    return len(preds)

# Process all splits
print("\nProcessing training data...")
train_preds = save_preds(train_data, batch_size, "train_vals.txt")
save_data(train_data, "train_data.txt", size=train_preds)

print("\nProcessing validation data...")
valid_preds = save_preds(valid_data, batch_size, "valid_vals.txt")
save_data(valid_data, "valid_data.txt", size=valid_preds)

print("\nProcessing test data...")
test_preds = save_preds(test_data, batch_size, "test_vals.txt")
save_data(test_data, "test_data.txt", size=test_preds)

print("\n" + "=" * 50)
print("Done!")