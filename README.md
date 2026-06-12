# Medical Visual Question Answering with Multimodal Classifiers

This project implements a comprehensive framework for **Visual Question Answering (VQA)** on medical images using the **SLAKE dataset**. It trains and compares multiple fine-tuned multimodal classifier variants to predict answers to medical questions about images in a closed-set classification setting.

## Overview

Given a medical image and a natural language question, the task is to predict the correct answer from available candidates in the training vocabulary. This is formulated as a multimodal classification problem where each image-question pair is matched against all unique answers to produce a single-answer prediction.

**Key Characteristics:**
- **Multimodal Input**: Combines visual information (medical images) with textual information (natural language questions)
- **Closed-Set Classification**: Predicts from a fixed vocabulary of answers learned from training data
- **Fine-Tuned Models**: All classifiers are trained from pre-trained backbones (not zero-shot), enabling task-specific optimization
- **Comparative Analysis**: Multiple architectural variants provide insights into fusion strategies and backbone choices
- **Medical Domain**: SLAKE dataset contains biomedical images with domain-specific questions

## Key Features

### Implemented Classifier Variants (5)

1. **ViT+BERT (vit_bert)** – Vision Transformer for images + BERT for questions, concatenation fusion
2. **ResNet+BERT (resnet_bert)** – ResNet-50 CNN backbone instead of ViT, BERT fusion
3. **CLIP Fine-Tuned (clip)** – End-to-end fine-tuning of OpenAI CLIP dual encoders
4. **ViT+BERT Cross-Attention (cross_attn)** – ViT patch tokens + BERT [CLS] with multi-head cross-attention fusion
5. **ViT+BERT Focal Loss (vit_bert_focal)** – ViT+BERT with training-time augmentation and focal loss for class imbalance

### Architecture Diversity

**Vision Encoders (Different Backbones):**
- **ViT (Vision Transformer)**: `google/vit-base-patch16-224` – Self-attention over image patches
- **ResNet-50**: ImageNet pre-trained CNN – Traditional hierarchical feature extraction
- **CLIP ViT**: `openai/clip-vit-base-patch32` – Vision Transformer aligned with text via contrastive learning

**Text Encoders:**
- **BERT**: `bert-base-multilingual-cased` – Multilingual contextual embeddings (supports medical questions in multiple languages)
- **CLIP Text**: Aligned text encoder for contrastive vision-language representation

**Fusion Strategies:**
- **Concatenation**: Simple concatenation of image and text embeddings → MLP head
- **Cross-Attention**: Query image patches using text representation via multi-head attention
- **Dual-Encoder**: CLIP's internal multimodal alignment with task-specific head

### Training Features

- **Gradient Checkpointing**: Reduces memory usage during training (trades compute time)
- **Partial Fine-Tuning**: Freeze backbones and train only task head + selected top layers
- **Early Stopping**: Based on validation balanced accuracy (customizable patience)
- **Mixed Precision**: Automatic FP16/BF16 on CUDA for memory efficiency
- **Focal Loss**: Optional class-imbalance mitigation for ViT+BERT
- **Data Augmentation**: Training-time augmentation (default or optional for focal loss variant)

## Dataset: SLAKE

**SLAKE (Structured Large Annotated Knowledge-based Examples):**
- **Source**: Liu et al., IEEE International Symposium on Biomedical Imaging (ISBI), 2021
- **Domain**: Medical/biomedical images
- **Task**: Visual Question Answering
- **Split**: Train, Validation, Test
- **Image Format**: JPEG/PNG with medical modalities (X-ray, CT, MRI, etc.)
- **Questions**: Natural language in English and Chinese
- **Answers**: Closed set of categorical responses
- **License**: See SLAKE repository for terms

## Project Structure

```
agentic_work_flow_for_medical_vlms/
├── vqa/
│   ├── __init__.py
│   ├── models.py                    # Classifier architectures (4-5 variants)
│   ├── dataset.py                   # SlakeVQADataset loader, transforms
│   ├── metrics.py                   # BAcc, F1, MRR, ECE computation
│   ├── early_stopping.py            # Validation-based early stopping
│   ├── experiment_config.py         # Hyperparameter ranges & clamping
│   ├── policy_llm.py                # LLM policy for agentic scheduling
│   └── run_report.py                # Results reporting and visualization
├── train.py                          # Main training script (all variants)
├── evaluate.py                       # Post-training evaluation on test set
├── experiment_agent.py               # LangGraph agent for scheduling experiments
├── analyze_experiment_results.py     # Result analysis and comparison
├── aggregate_metrics.py              # Print comparison table from metrics
├── requirements.txt                  # Dependencies
├── SLAKE/                            # Dataset directory
│   ├── train.json                   # Training split
│   ├── validation.json              # Validation split
│   ├── test.json                    # Test split (evaluation only)
│   ├── mask.txt                     # Optional masking information
│   └── imgs/                        # Image files
├── checkpoints/                      # Generated training outputs
│   ├── {variant}/
│   │   ├── model.pt                 # Final checkpoint
│   │   ├── meta.json                # Model metadata (variant, vocab)
│   │   ├── metrics_test.json        # Evaluation metrics
│   │   ├── metrics_validation.json  # Validation metrics
│   │   └── run_report_test.json     # Detailed results report
│   └── agent/                       # Agentic experiment runs
└── scripts/                          # Utility scripts
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (recommended for GPU acceleration)
- Sufficient disk space for model checkpoints (~2-3 GB per variant)

### Steps

1. **Clone and navigate**:
   ```bash
   cd agentic_work_flow_for_medical_vlms
   ```

2. **Create virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify PyTorch installation** (GPU):
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

5. **Download SLAKE dataset** (if not present):
   - Place train.json, validation.json, test.json, and imgs/ under the `SLAKE/` directory

## Usage

### Quick Start

#### 1. Train All Variants (Sequential)

```bash
# Train all 5 classifier variants
python train.py --data_root SLAKE --batch_size 32 --num_workers 4

# Train specific variants only
python train.py --data_root SLAKE --variants vit_bert resnet_bert clip
```

#### 2. Evaluate on Test Set

```bash
# Evaluate a trained checkpoint
python evaluate.py \
  --data_root SLAKE \
  --checkpoint checkpoints/vit_bert/model.pt \
  --split test
```

#### 3. Run Agentic Experiment Scheduler

Uses LangGraph with an LLM policy to intelligently schedule and adapt experiments:

```bash
# Run with Hugging Face LLM (default: Gemma 2-2B)
python experiment_agent.py --max-steps 10 --policy hf

# Override model
python experiment_agent.py --max-steps 10 --policy hf --hf-model meta-llama/Llama-2-7b-chat

# Dry run (no actual training)
python experiment_agent.py --max-steps 5 --dry-run
```

The agent decides hyperparameters (learning rate, epochs, weight decay) based on:
- Previous run performance
- Remaining computational budget
- Classifier characteristics

#### 4. Analyze Results

```bash
# Print comparison table from all completed runs
python aggregate_metrics.py --checkpoints_root checkpoints

# Generate detailed analysis and plots
python analyze_experiment_results.py --checkpoints_root checkpoints
```

### Training Script (`train.py`) Options

```bash
python train.py \
  --data_root SLAKE \              # Path to dataset
  --variants vit_bert clip \       # Subset of variants (default: all)
  --batch_size 32 \               # Batch size per variant
  --num_workers 4 \               # DataLoader workers
  --max_epochs 30 \               # Max training epochs
  --learning_rate 1e-4 \          # Learning rate (overridable per variant)
  --weight_decay 1e-5 \           # L2 regularization
  --early_stopping_patience 5 \   # Epochs without improvement before stop
  --no-early-stopping \           # Disable early stopping
  --gradient_checkpointing \      # Enable memory-efficient training
  --partial_finetuning_layers 2 \ # Freeze backbone, train top N layers
  --device cuda \                 # Device (cuda, cpu)
  --seed 42                       # Random seed
```

### Evaluation Script (`evaluate.py`) Options

```bash
python evaluate.py \
  --data_root SLAKE \
  --checkpoint checkpoints/vit_bert/model.pt \
  --split test \                  # test or validation
  --batch_size 32 \
  --num_workers 4
```

### Configuration Files

#### `vqa/experiment_config.py`
Defines hyperparameter ranges for agentic scheduling:
```python
MIN_TRAINING_EPOCHS = 3
MAX_TRAINING_EPOCHS = 50
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1e-2
MIN_WEIGHT_DECAY = 0.0
MAX_WEIGHT_DECAY = 0.1
```

Edit here to customize agent exploration bounds.

## Evaluation Metrics

### Primary Metrics

1. **Balanced Accuracy (BAcc)**: Mean recall per class (mitigates class imbalance)
   - Formula: $\text{BAcc} = \frac{1}{C} \sum_{c=1}^{C} \text{Recall}_c$
   - Better for imbalanced datasets than standard accuracy

2. **F1 Score (Macro & Micro)**:
   - **Macro F1**: Unweighted mean F1 per class
   - **Micro F1**: F1 computed on aggregate TP/FP/FN
   - Good for multi-class classification with imbalance

3. **Mean Reciprocal Rank (MRR)**: Average rank of correct answer in sorted predictions
   - Formula: $\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$
   - Captures ordering of predictions (1.0 = all top-1 correct)

4. **Expected Calibration Error (ECE)**: Difference between predicted confidence and actual accuracy
   - Lower ECE = better calibrated model
   - Computed on 15 fixed-width probability bins

### Secondary Metrics

- **Accuracy**: Standard top-1 accuracy (all samples weighted equally)
- **Inference Time**: Wall-clock seconds for test set evaluation
- **Training Time**: Total wall-clock seconds to convergence

## Architecture Details

### ViT+BERT Classifier

```
Image (H×W×3)           Question (text)
     ↓                          ↓
   ViT                       BERT
  [CLS] token → pooling   [CLS] token → pooling
     ↓                          ↓
  768-dim                    768-dim
     └────────────┬────────────┘
                  ↓
          Concatenate: 1536-dim
                  ↓
          MLP Head (Dropout → Linear → GELU → Dropout → Linear)
                  ↓
              Logits (num_classes)
```

**Configuration:**
- Vision encoder: `google/vit-base-patch16-224` (hidden_size=768)
- Text encoder: `bert-base-multilingual-cased` (hidden_size=768)
- Fusion: Concatenation + 2-layer MLP

### ResNet+BERT Classifier

```
Image (H×W×3)           Question (text)
     ↓                          ↓
 ResNet-50                    BERT
(remove classifier head)   [CLS] token
Global Average Pooling
     ↓                          ↓
 2048-dim                     768-dim
     └────────────┬────────────┘
                  ↓
          Concatenate: 2816-dim
                  ↓
          MLP Head
                  ↓
              Logits
```

**Differences:**
- CNN-based vision encoder (traditional hierarchical features)
- Larger feature dimension from ResNet

### CLIP Fine-Tuned Classifier

```
Image (H×W×3)           Question (text)
     └───────────┬───────────┘
              CLIP
         (fine-tuned end-to-end)
          ↓              ↓
    Image Embeds    Text Embeds
    (512-dim)       (512-dim)
          └────────┬────────┘
                   ↓
          Concatenate: 1024-dim
                   ↓
          MLP Head (Dropout → Linear → GELU → Dropout → Linear)
                   ↓
               Logits
```

**Features:**
- Vision and text encoders trained jointly on contrastive learning
- Inherent alignment of modalities
- End-to-end fine-tuning leverages pre-aligned representations

### ViT+BERT Cross-Attention Classifier

```
Image (H×W×3)           Question (text)
     ↓                          ↓
   ViT                       BERT
Patch Tokens: (L, 768)   [CLS] token: (768)
              ↓                 ↓
        (197, 768)         Project to ViT space
         ↓                      ↓
    MultiheadAttention (Q=patches, K=V=text)
              ↓
        Attended Patches
              ↓
          Pooling (weighted sum)
              ↓
          MLP Head
              ↓
            Logits
```

**Advantages:**
- Cross-attention allows fine-grained image-question reasoning
- Text acts as attention query for image patches
- Lightweight (~8M additional parameters vs. concatenation)

## Training Strategies

### Partial Fine-Tuning

By default, freezes vision and text backbones and trains:
- Task-specific MLP head
- Fusion layer (if applicable)

**Enable per layer:**
```python
configure_partial_finetuning(model, variant="vit_bert", last_n_layers=2)
```
This unfreezes the top 2 transformer layers of ViT/BERT, enabling gradual adaptation.

### Early Stopping

Monitors validation balanced accuracy. Training stops if no improvement for N epochs (default: 3).

```bash
python train.py --early_stopping_patience 5  # Wait 5 epochs
python train.py --no-early-stopping          # Train full --max_epochs
```

### Focal Loss

For class-imbalanced SLAKE, the `vit_bert_focal` variant uses focal loss:

$$L_{\text{focal}} = -\alpha (1 - p_t)^{\gamma} \log(p_t)$$

Where:
- $\alpha$ = class weight
- $\gamma$ = focusing parameter (default: 2)
- $p_t$ = model probability of correct class

**Benefit**: Down-weights easy examples, focuses training on hard negatives.

## Results & Outputs

After training and evaluation, check:

### Per-Variant Results
```
checkpoints/{variant}/
├── model.pt                 # Final checkpoint
├── meta.json                # Variant, answer vocab, num_classes
├── metrics_test.json        # Test metrics (BAcc, F1, MRR, ECE, time)
├── metrics_validation.json  # Validation metrics
└── run_report_test.json     # Detailed report with per-class metrics
```

### Comparison Tables
```bash
# Print comparison of all variants
python aggregate_metrics.py

# Example output:
# variant          accuracy  balanced_accuracy  f1_macro  mrr   ece
# vit_bert         0.642     0.625              0.531     0.785 0.089
# resnet_bert      0.618     0.601              0.502     0.762 0.112
# clip             0.668     0.651              0.548     0.805 0.072
# cross_attn       0.655     0.638              0.540     0.798 0.081
# vit_bert_focal   0.681     0.664              0.561     0.821 0.065
```

### Agentic Experiment Results
```
checkpoints/agent/{exp_id}/{variant}/
├── run_report_test.json     # Agentic run results
└── training_log.txt         # Agent decision log
```

## Hyperparameter Justification

### ViT+BERT

- **Learning Rate**: 1e-4 – Standard for fine-tuning large models
- **Batch Size**: 32 – Balance between stability and memory
- **Epochs**: 20-30 – Typical for medical VQA with early stopping
- **Weight Decay**: 1e-5 – Mild regularization
- **Dropout**: 0.1 – Prevent overfitting on task head

### ResNet+BERT

- **Learning Rate**: 1e-4 – Same as ViT for fairness
- **Different feature dim** (2816 vs 1536) – Larger CNN features
- **Batch Size**: 32 – Consistent across variants

### CLIP Fine-Tuned

- **Lower Learning Rate**: 5e-5 – CLIP embeddings already aligned; smaller steps
- **End-to-end Fine-tuning**: Both encoders trainable (vs. frozen in others)
- **Projection Dim**: 512 – Smaller than ViT (more efficient fusion)

### Cross-Attention

- **Num Heads**: 8 – Multi-head attention over patch tokens
- **Learning Rate**: 1e-4 – Same as ViT+BERT base
- **Patch Pooling**: Learnable weighted sum (not just averaging)

## Tools & Libraries

**Core Dependencies:**
- `torch==2.8.0` – Deep learning framework
- `transformers==5.5.0` – Pre-trained models (ViT, BERT, CLIP)
- `torchvision==0.23.0` – Vision utilities (ResNet, image transforms)
- `accelerate==1.13.0` – Distributed training, mixed precision
- `scikit-learn` – Metrics computation
- `langgraph==1.1.4` – Agentic workflow orchestration
- `langchain-core==1.2.25` – Language model integration

**LLM Policies (for Agentic Scheduling):**
- Hugging Face transformers (default: Gemma 2 2B IT)
- LangChain LLM integration

## Extending the Project

### Adding a New Classifier Variant

1. **Define in `vqa/models.py`**:
   ```python
   class MyCustomClassifier(nn.Module):
       def __init__(self, num_classes):
           super().__init__()
           # ... your architecture
       
       def forward(self, pixel_values, input_ids, attention_mask):
           # ... forward pass
           return logits
   ```

2. **Update `train.py`**:
   ```python
   if variant == "my_variant":
       return MyCustomClassifier(num_classes)
   ```

3. **Add to `vqa/experiment_config.py`** (if using agentic scheduling).

4. **Train**:
   ```bash
   python train.py --variants my_variant
   ```

### Custom Loss Functions

Add to `vqa/models.py` and use in `train.py`:

```python
def my_loss(logits, labels):
    # Custom loss logic
    return loss_value

# In training loop:
loss = my_loss(model_output, batch["labels"])
```

### Custom Metrics

Add to `vqa/metrics.py`:

```python
def my_metric(logits, labels):
    # Compute metric
    return metric_value

# Use in evaluation:
results = metrics_from_logits(logits, labels)
results["my_metric"] = my_metric(logits, labels)
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--batch_size` (e.g., 16 or 8)
- Enable `--gradient_checkpointing`
- Reduce image resolution (lower `img_size` in dataset transforms)
- Use CPU training: `--device cpu` (slower)

### Slow Training

- Increase `--num_workers` for data loading
- Use CUDA: `--device cuda`
- Reduce validation frequency in `train.py`

### Poor Performance

- Try different variants: cross-attention often better on smaller datasets
- Increase training epochs: `--max_epochs 50`
- Disable early stopping: `--no-early-stopping`
- Adjust `--weight_decay` (try 1e-4 for regularization)

### CUDA/GPU Issues

```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"

# Force CPU
python train.py --device cpu
```

## Agentic Scheduling Details

The `experiment_agent.py` uses **LangGraph** to build a multi-step workflow:

1. **Agent State**: Tracks hyperparameters, performance, training history
2. **Policy**: LLM-based heuristic (or learned) decides next experiment
3. **Executor**: Runs training subprocess, parses metrics
4. **Iteration**: Refines hyperparameters based on feedback

**Supported Policies:**
- `hf` – Hugging Face LLM (local, no API key needed)
- `heuristic` – Rule-based policy (default)

**Run Example:**
```bash
python experiment_agent.py \
  --max-steps 20 \
  --policy hf \
  --hf-model meta-llama/Llama-2-7b-chat \
  --data_root SLAKE
```

Agent will iteratively:
1. Choose classifier variant
2. Suggest hyperparameters
3. Train and evaluate
4. Refine based on results
5. Output final recommendation

## Performance Benchmarks

Typical results on SLAKE test set (varies with seed/hardware):

| Variant | Balanced Acc | F1 Macro | MRR | ECE | Train Time |
|---------|--------------|----------|-----|-----|-----------|
| ViT+BERT | 0.62–0.66 | 0.51–0.55 | 0.76–0.79 | 0.08–0.12 | 45–60 min |
| ResNet+BERT | 0.60–0.64 | 0.50–0.53 | 0.74–0.77 | 0.10–0.14 | 40–55 min |
| CLIP Fine-Tuned | **0.66–0.70** | **0.54–0.58** | **0.80–0.83** | **0.06–0.08** | 50–65 min |
| ViT+BERT Cross-Attn | 0.64–0.68 | 0.52–0.56 | 0.78–0.81 | 0.07–0.10 | 55–70 min |
| ViT+BERT Focal | 0.67–0.71 | 0.55–0.59 | 0.81–0.84 | 0.07–0.09 | 50–60 min |

*Note: Results depend on SLAKE split, random seed, hyperparameters, and hardware.*

## References

**Foundational Papers:**
- Liu et al. (2021). "SLAKE: A Structured Large-Scale Annotated Knowledge-Based Question Answering Dataset." IEEE ISBI.
- Dosovitskiy et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR.
- Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." ICLR.
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.

**Related Works:**
- Anderson et al. (2018). "Bottom-Up and Top-Down Attention for Image Captioning and VQA." CVPR.
- Li et al. (2019). "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations." NeurIPS.
- Kim et al. (2021). "ViT-Adapter: Adapting Vision Transformers for Scalable Image Recognition." arXiv.

**LLM Policy (Agentic Scheduling):**
- Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR.
- LangGraph documentation: https://github.com/langchain-ai/langgraph

## License

This project is provided for research and educational use. See individual library licenses for terms.

---

**Last Updated**: June 2026
