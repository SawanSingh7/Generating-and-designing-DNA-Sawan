import numpy as np
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lib

parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int, default=100, help="How long the sequences in the dataset are")
parser.add_argument('--vocab', type=str, default="dna_nt_only", help="Which vocabulary to use. Options are 'dna', 'rna', 'dna_nt_only', and 'rna_nt_only'.")
parser.add_argument('--bg', type=str, default="CG", help="Characters which are allowed to appear in the background")
parser.add_argument('--motif', type=str, default="TATA", help="Foreground motif")
parser.add_argument('--dataset_size', type=int, default=50000, help="Number of data examples to create")
parser.add_argument('--out', type=str, default="../data", help="Folder to save the output file in.")
args = parser.parse_args()

bg_chars = args.bg
motif = args.motif
charmap, rev_charmap = lib.dna.get_vocab(args.vocab)
vocab_len = len(charmap)

data = np.random.choice(list(bg_chars), size=[args.dataset_size, args.seq_len], replace=True)
motif_locs = args.seq_len - len(motif) + 1
motif_placement = np.random.choice(motif_locs, args.dataset_size)
seqs = []
for idx, place in enumerate(motif_placement):
  new_seq = data[idx].tolist()
  new_seq[place:place + len(motif)] = motif
  seqs.append("".join(new_seq))

save_loc = os.path.join(os.path.expanduser(args.out))
os.makedirs(save_loc, exist_ok=True)
with open(os.path.join(save_loc, "toy_single_motif_data_{}x{}x{}".format(args.dataset_size, args.seq_len, vocab_len)), "w") as f:
  f.write("\n".join(seqs))

print("Done")