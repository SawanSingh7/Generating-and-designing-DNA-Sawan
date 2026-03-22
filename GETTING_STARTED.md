# DNA Generation & Design Repository - Getting Started Guide

## Quick Start

The repository is now fully functional and can be used immediately without external data files!

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# All packages needed: TensorFlow 2.16+, NumPy, Matplotlib, Pandas, editdistance
```

### 2. Generate Sample Data (No External Files Needed!)

```bash
# Generate toy DNA sequences with start/stop codons
python scripts/make_toy_coding_seqs_data.py

# Generate DNA with embedded motif patterns
python scripts/make_toy_single_motif_dataset.py

# Create exon datasets (auto-generates synthetic exons if files missing)
python scripts/make_exon_datasets.py --data_loc data

# Calculate edit distances (auto-generates sequences if missing)
python scripts/edit_distance.py --set test
```

### 3. Train Models

```bash
# Train CNN predictor (auto-generates toy data)
python scripts/train_predictor_tf2.py --num_epochs 2

# Train GAN for DNA generation (auto-generates toy data)
python scripts/train_gan_tf2.py --train_iters 10
```

### 4. Optimize Sequences

```bash
# Single-objective optimization (maximize predicted binding)
python scripts/plug_and_play.py --iterations 100 --target max

# Multi-objective optimization (max binding, min specificity)
python scripts/plug_and_play_multi.py \
    --iterations 100 \
    --target1 max \
    --target2 min \
    --target1_scale 1.5

# Auto-minimizes sequences instead
python scripts/plug_and_play.py --iterations 50 --target min
```

## Key Features

### ✅ Auto-Generation of Missing Data
Scripts automatically create synthetic data when external files are missing:
- **edit_distance.py**: Generates DNA sequences
- **make_exon_datasets.py**: Generates exon annotations
- **plug_and_play.py**: Generates generator and predictor models
- **plug_and_play_multi.py**: Generates dual predictor models

### ✅ Sensible Defaults
- `--data_loc`: Defaults to `../data` (created if missing)
- `--log_dir`: Defaults to `../logs`
- All models save automatically with timestamps
- Batch sizes, learning rates, defaults all pre-tuned

### ✅ TensorFlow 2.x Throughout
- All scripts use eager execution
- GradientTape for custom training loops
- Keras Models for all architectures
- `.keras` format for model persistence

### ✅ Reproducible Results
- Global random seeds settable with `--seed`
- All models save to disk automatically
- Checkpoints saved at regular intervals
- Plots generated for visualization

## Directory Structure

```
.
├── README.md                    # Original repository README
├── GETTING_STARTED.md          # This file
├── LATEST_UPDATES.md           # Recent changes summary
├── requirements.txt            # Python dependencies
│
├── lib/
│   ├── __init__.py            # Module initialization
│   ├── dna.py                 # DNA utilities
│   ├── explicit.py            # Explicit models
│   ├── models.py              # Model definitions
│   └── utils.py               # Helper utilities
│
├── scripts/                    # Main executable scripts
│   ├── train_predictor.py     # CNN binding predictor training
│   ├── train_gan_tf2.py       # GAN for sequence generation
│   ├── train_predictor_tf2.py # Alternative CNN trainer
│   │
│   ├── make_toy_coding_seqs_data.py      # Generate toy sequences
│   ├── make_toy_single_motif_dataset.py  # Generate motif sequences
│   ├── make_ground_truth.py              # Create train/valid/test splits
│   ├── make_exon_datasets.py             # Process exon data
│   │
│   ├── edit_distance.py                  # Sequence similarity metrics
│   ├── test_ground_truth.py              # Evaluate predictor
│   │
│   ├── plug_and_play.py                  # Single-objective optimization
│   ├── plug_and_play_multi.py            # Multi-objective optimization
│   │
│   ├── build_baselines.py                # Random sequence baseline
│   ├── build_nogen_graph.py              # Generator-free baseline
│   ├── build_singlesoftmax_graph.py      # Softmax baseline
│   └── build_max_match_graph.py          # Pattern matching baseline
│
├── data/                       # Data directory (auto-created)
│   ├── train_data.txt         # Training sequences
│   ├── valid_data.txt         # Validation sequences
│   ├── test_data.txt          # Test sequences
│   └── *.txt                  # Generated data files
│
└── logs/                       # Logs directory (auto-created)
    ├── predictor/             # Predictor training logs
    ├── gan_test/              # GAN training logs
    ├── pp_test/               # Optimization results
    └── ...                    # Other experiment results
```

## Common Workflows

### Workflow 1: Quick Test (No External Data)
```bash
# 1. Generate toy data
python scripts/make_toy_single_motif_dataset.py

# 2. Train predictor briefly
python scripts/train_predictor_tf2.py --num_epochs 2

# 3. Optimize sequences
python scripts/plug_and_play.py --iterations 20 --batch_size 32
```
**Time: ~5 minutes** ✅

### Workflow 2: Full GAN Pipeline
```bash
# 1. Generate training data
python scripts/make_toy_coding_seqs_data.py

# 2. Train GAN
python scripts/train_gan_tf2.py --train_iters 100

# 3. Train predictor
python scripts/train_predictor_tf2.py --num_epochs 5

# 4. Optimize with trained models
python scripts/plug_and_play.py \
    --generator logs/gan_test/*/generator.keras \
    --predictor logs/predictor/*/predictor.keras \
    --iterations 100
```
**Time: ~30-60 minutes** ⏱️

### Workflow 3: Multi-Objective Design
```bash
# Design sequences that maximize binding to target but minimize specificity
python scripts/plug_and_play_multi.py \
    --predictor1 binding_model.keras \
    --predictor2 specificity_model.keras \
    --target1 max \
    --target2 min \
    --target1_scale 2.0 \
    --iterations 100
```

### Workflow 4: Exon Analysis
```bash
# Create exon datasets (auto-generates synthetic if real files missing)
python scripts/make_exon_datasets.py --data_loc data

# Calculate edit distances
python scripts/edit_distance.py --set train
```

## Command Line Reference

### Data Generation Scripts

```bash
# Toy coding sequences (with start/stop codons)
python scripts/make_toy_coding_seqs_data.py \
    --output_dir data \
    --num_sequences 1000 \
    --seq_len 36

# Toy single motif (sequences with embedded pattern)
python scripts/make_toy_single_motif_dataset.py \
    --num_pos 500 \
    --num_neg 500 \
    --seq_len 36

# Exon datasets
python scripts/make_exon_datasets.py \
    --data_loc data \
    --valid_frac 0.1 \
    --test_frac 0.1

# Edit distance computation
python scripts/edit_distance.py \
    --set train \
    --data_dir data
```

### Training Scripts

```bash
# Train CNN predictor
python scripts/train_predictor_tf2.py \
    --data_loc data \
    --num_epochs 10 \
    --batch_size 64 \
    --learning_rate 0.001

# Train GAN
python scripts/train_gan_tf2.py \
    --data_loc data \
    --train_iters 1000 \
    --batch_size 64 \
    --latent_dim 100 \
    --learning_rate 0.0002
```

### Optimization Scripts

```bash
# Single-objective (max/min/specific value)
python scripts/plug_and_play.py \
    --generator generator.keras \
    --predictor predictor.keras \
    --target max \
    --iterations 100 \
    --batch_size 64 \
    --step_size 0.1

# Multi-objective
python scripts/plug_and_play_multi.py \
    --generator generator.keras \
    --predictor1 binding.keras \
    --predictor2 specificity.keras \
    --target1 max \
    --target2 min \
    --target1_scale 1.5 \
    --iterations 200 \
    --batch_size 64
```

## Output Files

### After Training
- `logs/*/predictor.keras` - Trained CNN model
- `logs/*/generator.keras` - Trained GAN generator
- `logs/*/training_history.json` - Loss/accuracy over time
- `logs/*/checkpoint_*.keras` - Periodic checkpoints

### After Optimization
- `logs/pp_test/*/samples/` - Generated sequences
- `logs/pp_test/*/scores_opt.png` - Optimization progress plot
- `logs/pp_test/*/scores.txt` - Score history

### Data Files Generated
- `data/train_data.txt` - Training sequences
- `data/valid_data.txt` - Validation sequences
- `data/test_data.txt` - Test sequences
- `data/*_ann.txt` - Annotations (if applicable)

## Reproducibility

All scripts support `--seed` for reproducible results:

```bash
# Seeds all random operations for reproducible training
python scripts/train_predictor_tf2.py --seed 42

# Same seed ensures identical optimization results
python scripts/plug_and_play.py --seed 42
```

## Troubleshooting

### Issue: "FileNotFoundError: No such file or directory"
**Solution**: Scripts auto-generate missing files! If this happens:
```bash
mkdir -p data
python scripts/make_toy_single_motif_dataset.py
```

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "CUDA not available" warning
**Solution**: Normal! CPU training works fine, just slower. GPU not required.

### Issue: Models are very large (100+ MB)
**Solution**: Compress with gzip:
```bash
gzip logs/*/generator.keras
```

## Performance Notes

### Expected Training Times (CPU)
- Toy data generation: < 1 second
- CNN training (10 epochs): 2-5 minutes
- GAN training (100 iterations): 10-30 minutes
- Optimization (100 iterations): 5-15 minutes

### Memory Usage
- Typical models: 10-50 MB
- Training batches: 100-500 MB
- Total: Usually < 2 GB

## Citation

If you use this repository, please cite:
```
Killoran et al. "Generating and designing DNA with deep generative models"
```

## Support

For issues or questions:
1. Check this guide first
2. Review tool output carefully - they're informative!
3. Try `--seed 42` for reproducibility
4. Reduce `--batch_size` if running out of memory

## What's New in This Version

✅ Full TensorFlow 2.x compatibility
✅ Auto-generation of synthetic data/models
✅ Sensible default arguments
✅ Removed all TensorFlow 1.x code
✅ Reproducible random seeds
✅ Clean error messages
✅ Portable codespace-ready setup

---

**Happy sequence designing! 🧬**
