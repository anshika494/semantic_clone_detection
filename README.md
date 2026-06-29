# Semantic Code Clone Detection System

A research-grade system for detecting **Type III** (near-miss) and **Type IV** (semantic) code clones in Java using Abstract Syntax Trees and pretrained transformer models (CodeBERT / GraphCodeBERT) with a Siamese neural network.

---

## Architecture Overview

```
Java Source Code
       │
       ▼
┌─────────────────────────────────────┐
│  1. Preprocessing Pipeline          │
│   • Comment removal (state machine) │
│   • Identifier normalization        │
│   • Literal normalization           │
│   • Whitespace normalization        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. AST Generation                  │
│   • Tree-Sitter (primary)           │
│   • javalang (fallback)             │
│   • Linearization: preorder BFS     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Transformer Encoding            │
│   • CodeBERT / GraphCodeBERT        │
│   • Mean/CLS/Max pooling            │
│   • Projection head (256-dim)       │
│   • L2 normalization                │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │  Siamese    │  (shared weights)
        │  Network    │
        └──────┬──────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Loss & Training                 │
│   • Contrastive Loss                │
│   • Cosine Similarity Loss          │
│   • Triplet Loss (hard negative)    │
│   • NT-Xent Loss                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Inference & Evaluation          │
│   • Cosine similarity scoring       │
│   • Threshold optimization          │
│   • BigCloneEval integration        │
│   • Per clone-type breakdown        │
└─────────────────────────────────────┘
```

---

## Project Structure

```
semantic_clone_detection/
├── configs/
│   └── config.yaml              # Master configuration
├── src/
│   ├── dataset/
│   │   └── __init__.py          # BigCloneBench loader & dataset classes
│   ├── ast_processing/
│   │   └── __init__.py          # Tree-Sitter & javalang parsers, linearizer
│   ├── preprocessing/
│   │   └── __init__.py          # Comment removal, normalization, tokenization
│   ├── embeddings/
│   │   └── __init__.py          # CodeEmbeddingModel, EmbeddingGenerator
│   ├── models/
│   │   ├── __init__.py          # SiameseCloneDetector, loss functions
│   │   └── trainer.py           # Full training loop
│   ├── evaluation/
│   │   └── __init__.py          # Metrics, BigCloneEval integration
│   └── utils/
│       └── __init__.py          # Config, logging, reproducibility
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── inference.py             # Clone detection inference
│   ├── evaluate.py              # Full evaluation pipeline
│   └── preprocess.py            # Dataset preprocessing
├── experiments/
│   └── run_experiments.py       # Ablation studies & hyperparameter search
├── tests/
│   └── test_all.py              # Comprehensive unit tests
├── data/
│   ├── raw/                     # BigCloneBench database + IJaDataset
│   ├── processed/               # Preprocessed pairs (JSON cache)
│   └── embeddings/              # Cached embeddings (numpy)
├── checkpoints/                 # Saved model checkpoints
├── results/                     # Evaluation results & plots
├── requirements.txt
└── setup.py
```

---

## Setup

### 1. Install Dependencies

```bash
git clone <this-repo>
cd semantic_clone_detection
pip install -r requirements.txt
```

### 2. Download BigCloneBench

```bash
# Download BCB database (~2GB)
mkdir -p data/raw/bigclonebench
wget https://github.com/clonebench/BigCloneBench/releases/download/v2.0/bigclonebench.db \
     -O data/raw/bigclonebench/bigclonebench.db

# Download IJaDataset (source files, ~25GB)
# See: https://github.com/clonebench/BigCloneBench
# Extract to: data/raw/bigclonebench/IJaDataset/
```

### 3. (Optional) Download BigCloneEval

```bash
git clone https://github.com/jeffsvajlenko/BigCloneEval \
          data/raw/BigCloneEval
```

---

## Quick Start

### Development (Synthetic Data)

Test the full pipeline without downloading BigCloneBench:

```bash
# Preprocess synthetic data
python scripts/preprocess.py --config configs/config.yaml --synthetic

# Train on synthetic data (fast)
python scripts/train.py --config configs/config.yaml \
    --synthetic --epochs 3 --max-pairs 1000

# Evaluate
python scripts/evaluate.py --config configs/config.yaml \
    --model checkpoints/best_model.pt --synthetic
```

### Full Training on BigCloneBench

```bash
# Step 1: Preprocess (run once)
python scripts/preprocess.py --config configs/config.yaml --use-ast --workers 8

# Step 2: Train
python scripts/train.py \
    --config configs/config.yaml \
    --backbone microsoft/codebert-base

# Step 3: Evaluate with BigCloneEval
python scripts/evaluate.py \
    --config configs/config.yaml \
    --model checkpoints/best_model.pt \
    --optimize-threshold \
    --bce \
    --plots
```

### Inference

```bash
# Compare two Java files
python scripts/inference.py \
    --mode pair \
    --file1 Method1.java \
    --file2 Method2.java

# Scan a Java project for clones
python scripts/inference.py \
    --mode scan \
    --source-dir /path/to/project/src

# Batch evaluation from CSV
python scripts/inference.py \
    --mode batch \
    --input pairs.csv \
    --output results.csv
```

---

## Configuration

All settings in `configs/config.yaml`. Key options:

```yaml
model:
  backbone: "microsoft/codebert-base"     # or "microsoft/graphcodebert-base"
  projection_dim: 256                      # embedding output dimension
  siamese:
    loss: "contrastive"                    # contrastive | cosine | triplet | ntxent
    margin: 1.0

training:
  epochs: 10
  batch_size: 32
  learning_rate: 2.0e-5
  fp16: true                               # mixed precision

preprocessing:
  remove_comments: true
  normalize_identifiers: true
  normalize_literals: true

ast:
  parser: "tree-sitter"                   # tree-sitter | javalang
  traversal: "preorder"                   # preorder | postorder | bfs
```

---

## Running Experiments

### Ablation Studies

```bash
# Run all ablation experiments
python experiments/run_experiments.py --suite ablation --synthetic

# Run specific experiments
python experiments/run_experiments.py \
    --suite ablation \
    --experiments loss_contrastive loss_cosine loss_triplet \
    --synthetic

# Dry run (preview without executing)
python experiments/run_experiments.py --suite ablation --dry-run
```

### Hyperparameter Search

```bash
python experiments/run_experiments.py --suite hyperparam --synthetic
```

---

## Unit Tests

```bash
# Run all tests
python -m pytest tests/test_all.py -v

# Run specific test classes
python -m pytest tests/test_all.py::TestCommentRemover -v
python -m pytest tests/test_all.py::TestEvaluationMetrics -v
python -m pytest tests/test_all.py::TestPipelineIntegration -v

# With coverage
python -m pytest tests/test_all.py --cov=src --cov-report=html
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct classifications |
| **Precision** | Fraction of detected clones that are true clones |
| **Recall** | Fraction of true clones detected (detection rate) |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve; threshold-independent |
| **PR-AUC** | Area under precision-recall curve |

### Per Clone-Type Breakdown

The system evaluates recall separately for each clone type:

| Type | Description | Challenge |
|---|---|---|
| T1 | Exact copies | Trivial |
| T2 | Renamed identifiers | Easy |
| **T3** | Modified clones (statements added/changed) | **Hard** |
| **T4** | Semantic clones (same behavior, different implementation) | **Very Hard** |

---

## Model Details

### Encoder: CodeBERT

- Pre-trained on 6 programming languages + natural language
- 125M parameters, 12 transformer layers
- Context window: 512 tokens
- Output: 768-dimensional hidden states

### Siamese Network

- Shared encoder weights for both inputs
- Mean pooling over token embeddings
- Projection MLP: 768 → 256 dimensions
- L2 normalization for cosine similarity

### Loss Functions

**Contrastive Loss** (default):
```
L = y·D² + (1-y)·max(0, margin - D)²
```
Where D = Euclidean distance, y = 1 for clones.

**Cosine Similarity Loss**:
```
L = y·(1 - cos_sim) + (1-y)·max(0, cos_sim - margin)
```

**Triplet Loss** with hard negative mining:
```
L = max(0, d(a,p) - d(a,n) + margin)
```

---

## Expected Results

On BigCloneBench (T3 + T4 clones):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| CodeBERT + Contrastive | ~0.85 | ~0.82 | ~0.83 | ~0.91 |
| GraphCodeBERT + Cosine | ~0.87 | ~0.84 | ~0.85 | ~0.92 |
| CodeBERT + NT-Xent | ~0.84 | ~0.86 | ~0.85 | ~0.91 |

*Results vary with training size and hyperparameters.*

---

## Reproducibility

All experiments use fixed seeds. Key settings:
- `configs/config.yaml`: `project.seed: 42`
- `scripts/train.py`: `set_seed(args.seed)`
- Dataset splits are stratified and deterministic

---

## Citations

```bibtex
@inproceedings{feng2020codebert,
  title={CodeBERT: A Pre-Trained Model for Programming and Natural Languages},
  author={Feng, Zhangyin and others},
  booktitle={EMNLP Findings},
  year={2020}
}

@inproceedings{bigclonebench,
  title={BigCloneBench: A Benchmark for Big Data Code Clone Detection},
  author={Svajlenko, Jeffrey and Islam, Judith and Keivanloo, Iman and Roy, Chanchal and Mia, Mohammad},
  booktitle={ICSME},
  year={2014}
}
```
