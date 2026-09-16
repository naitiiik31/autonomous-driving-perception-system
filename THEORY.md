# 📖 Deep Learning Theory — Complete Reference Guide
## Autonomous Driving & Object Detection with YOLO

> **A comprehensive theoretical foundation covering every concept used in this project,
> from neural network basics to advanced YOLO object detection.**

---

## Table of Contents

1. [Introduction to Deep Learning](#1-introduction-to-deep-learning)
2. [Neural Networks — The Building Blocks](#2-neural-networks--the-building-blocks)
3. [Training Deep Networks](#3-training-deep-networks)
4. [Convolutional Neural Networks (CNNs)](#4-convolutional-neural-networks-cnns)
5. [Object Detection — Problem Formulation](#5-object-detection--problem-formulation)
6. [YOLO — You Only Look Once](#6-yolo--you-only-look-once)
7. [YOLO v2 — Better, Faster, Stronger](#7-yolo-v2--better-faster-stronger)
8. [DarkNet-19 — The Backbone Network](#8-darknet-19--the-backbone-network)
9. [Anchor Boxes & Bounding Box Regression](#9-anchor-boxes--bounding-box-regression)
10. [Loss Function — Multi-Task Learning](#10-loss-function--multi-task-learning)
11. [Non-Maximum Suppression (NMS)](#11-non-maximum-suppression-nms)
12. [Intersection over Union (IoU)](#12-intersection-over-union-iou)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Transfer Learning & Pre-trained Models](#14-transfer-learning--pre-trained-models)
15. [Autonomous Driving Context](#15-autonomous-driving-context)
16. [Mathematical Notation Reference](#16-mathematical-notation-reference)
17. [Glossary](#17-glossary)

---

## 1. Introduction to Deep Learning

### 1.1 What is Deep Learning?

Deep Learning is a subset of Machine Learning that uses **multi-layered neural networks** (deep neural networks) to automatically learn hierarchical feature representations from data. Unlike traditional machine learning where features are hand-engineered, deep learning **learns features directly from raw data**.

```
Traditional ML:  Raw Data → Hand-crafted Features → Classifier → Output
Deep Learning:   Raw Data → [Feature Learning + Classification] → Output
                              └── Learned automatically by the network
```

### 1.2 Why Deep Learning Works

Three factors converged to make deep learning the dominant paradigm:

1. **Big Data**: Internet-scale datasets (ImageNet: 14M images, COCO: 330K images)
2. **Compute Power**: GPU parallelism (NVIDIA CUDA, TPUs)
3. **Algorithmic Advances**: Better architectures, optimization, and regularization

### 1.3 The Universal Approximation Theorem

> *A neural network with a single hidden layer containing a finite number of neurons
> can approximate any continuous function on compact subsets of ℝⁿ.*

This means neural networks are theoretically capable of learning **any mapping** from inputs to outputs — the challenge is finding the right weights efficiently.

---

## 2. Neural Networks — The Building Blocks

### 2.1 The Perceptron (Single Neuron)

The fundamental computational unit:

```
Inputs: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b

Linear combination:  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
Activation:          a = σ(z)
```

### 2.2 Multi-Layer Perceptrons (MLPs)

Stacking neurons in layers creates a Multi-Layer Perceptron:

```
Input Layer    Hidden Layer(s)    Output Layer
   x₁ ──────→ h₁ ──────────────→ y₁
   x₂ ──────→ h₂ ──────────────→ y₂
   x₃ ──────→ h₃                 
              h₄

Forward propagation:
  z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
  a⁽ˡ⁾ = σ(z⁽ˡ⁾)
```

### 2.3 Activation Functions in Detail

#### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
f'(x) = { 1, if x > 0
         { 0, if x ≤ 0

Advantages:
✅ No vanishing gradient for positive values
✅ Computationally efficient
✅ Sparse activation (biological plausibility)

Disadvantage:
❌ "Dying ReLU" problem — neurons with negative input get zero gradient permanently
```

#### Leaky ReLU (Used in DarkNet-19)
```
f(x) = { x,      if x > 0
        { αx,     if x ≤ 0     (α = 0.1 in YOLO)

f'(x) = { 1,      if x > 0
         { α,      if x ≤ 0

✅ Prevents dying neurons by allowing small negative gradients
✅ Used throughout the DarkNet backbone in this project
```

#### Sigmoid
```
σ(x) = 1 / (1 + e⁻ˣ)

σ'(x) = σ(x) · (1 - σ(x))

Range: (0, 1)
Used in YOLO for:
- Box center prediction (tx, ty) → constrains to grid cell
- Objectness confidence → probability interpretation
```

#### Softmax
```
softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ

Properties:
- Outputs sum to 1 (valid probability distribution)
- Used for class probability prediction in YOLO
- Assumes mutually exclusive classes
```

### 2.4 Weight Initialization

Proper initialization is critical for training deep networks:

| Method | Formula | Best For |
|--------|---------|----------|
| Xavier/Glorot | W ~ N(0, 2/(nᵢₙ + nₒᵤₜ)) | Sigmoid, Tanh |
| He/Kaiming | W ~ N(0, 2/nᵢₙ) | ReLU, Leaky ReLU |
| Uniform | W ~ U(-√(6/(nᵢₙ+nₒᵤₜ)), √(6/(nᵢₙ+nₒᵤₜ))) | General |

---

## 3. Training Deep Networks

### 3.1 Loss Functions

The loss function quantifies how wrong the model's predictions are:

**Mean Squared Error (Used in YOLO v2):**
```
L_MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

**Binary Cross-Entropy:**
```
L_BCE = -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Categorical Cross-Entropy:**
```
L_CCE = -(1/n) Σᵢ Σc yᵢc log(ŷᵢc)
```

### 3.2 Gradient Descent Variants

```
Batch GD:       θ = θ - α · (1/m) Σᵢ ∇θL(xᵢ, yᵢ)    [All samples]
Stochastic GD:  θ = θ - α · ∇θL(xᵢ, yᵢ)               [One sample]
Mini-batch GD:  θ = θ - α · (1/B) Σᵢ∈B ∇θL(xᵢ, yᵢ)   [B samples]
```

### 3.3 Advanced Optimizers

#### Adam (Adaptive Moment Estimation)
```
Used in training YOLO:

mₜ = β₁ · mₜ₋₁ + (1 - β₁) · gₜ         (1st moment - momentum)
vₜ = β₂ · vₜ₋₁ + (1 - β₂) · gₜ²        (2nd moment - RMSprop)

m̂ₜ = mₜ / (1 - β₁ᵗ)                      (bias correction)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θₜ = θₜ₋₁ - α · m̂ₜ / (√v̂ₜ + ε)

Hyperparameters:
- α = 0.001 (learning rate)
- β₁ = 0.9 (momentum decay)
- β₂ = 0.999 (variance decay)
- ε = 10⁻⁸ (numerical stability)
```

### 3.4 Regularization Techniques

| Technique | How It Works | Effect |
|-----------|-------------|--------|
| **L2 Regularization** | Add λ‖w‖² to loss | Smaller weights, smoother models |
| **Dropout** | Randomly zero activations | Prevents co-adaptation |
| **Batch Normalization** | Normalize layer inputs | Faster training, regularization |
| **Data Augmentation** | Transform training images | More training diversity |
| **Early Stopping** | Stop when val loss increases | Prevents overfitting |

### 3.5 Batch Normalization (Critical for YOLO)

```
Input: Values x over a mini-batch B = {x₁, ..., xₘ}

1. Mini-batch mean:     μB = (1/m) Σᵢ xᵢ
2. Mini-batch variance: σ²B = (1/m) Σᵢ (xᵢ - μB)²
3. Normalize:           x̂ᵢ = (xᵢ - μB) / √(σ²B + ε)
4. Scale and shift:     yᵢ = γ · x̂ᵢ + β    (learnable parameters)

Benefits for YOLO:
✅ +2% mAP improvement over YOLO v1
✅ Allows higher learning rates
✅ Reduces sensitivity to initialization
✅ Acts as regularization (reduces need for dropout)
```

---

## 4. Convolutional Neural Networks (CNNs)

### 4.1 The Convolution Operation

A 2D convolution slides a kernel (filter) across the input:

```
For input I of size (H, W) and kernel K of size (k, k):

Output(i, j) = Σₘ₌₀ᵏ⁻¹ Σₙ₌₀ᵏ⁻¹ I(i+m, j+n) · K(m, n)
```

#### Output Size Calculation
```
Output Height = ⌊(H - k + 2p) / s⌋ + 1
Output Width  = ⌊(W - k + 2p) / s⌋ + 1

Where: H, W = input size
       k = kernel size
       p = padding
       s = stride
```

### 4.2 Multi-Channel Convolution

Real images have multiple channels (RGB = 3 channels):

```
Input:  (H, W, Cᵢₙ)
Filter: (k, k, Cᵢₙ) × Cₒᵤₜ filters
Output: (H', W', Cₒᵤₜ)

Each output channel = sum over all input channels convolved with filter
```

### 4.3 1×1 Convolutions (Network-in-Network)

Used extensively in YOLO for dimensionality reduction:

```
Input:  (H, W, 512)
1×1 Conv with 256 filters
Output: (H, W, 256)

Effect: Reduces channels without changing spatial dimensions
        Acts as a fully-connected layer across channels
        Adds non-linearity (with activation function)
```

### 4.4 Receptive Field

The receptive field is the region of the input that influences a single output neuron:

```
Layer 1 (3×3 conv): Receptive field = 3×3
Layer 2 (3×3 conv): Receptive field = 5×5
Layer 3 (3×3 conv): Receptive field = 7×7

With MaxPool (2×2, stride 2) between layers:
Layer 1 → Pool → Layer 2: Receptive field = 10×10

DarkNet-19 final layer: Receptive field covers most of the input image
→ This is why YOLO can see the "whole picture"
```

### 4.5 Feature Map Hierarchy

```
Layer Depth → Feature Complexity

Conv 1-3    → Edges, gradients, colors
              ┌──┐ ┌──┐ ╱╲
              │  │ │╱ │ ╱  ╲
              └──┘ └──┘ ╲  ╱

Conv 4-7    → Textures, patterns
              ┌──────┐
              │▓░▓░▓░│
              │░▓░▓░▓│
              └──────┘

Conv 8-13   → Object parts
              ┌──────┐
              │ 🔵🔵 │  (headlights)
              │──────│  (bumper)
              │ ○  ○ │  (wheels)
              └──────┘

Conv 14-19  → Complete objects
              🚗 🚌 🚶
```

---

## 5. Object Detection — Problem Formulation

### 5.1 Tasks in Computer Vision

```
Classification:     "Is this a car?"          → P(car) = 0.95
Localization:       "Where is the car?"       → [x, y, w, h]
Object Detection:   "What and where?"         → Multiple [class, x, y, w, h, conf]
Semantic Seg:       "Per-pixel classification" → Label each pixel
Instance Seg:       "Separate each object"     → Mask per instance
```

### 5.2 Detection Output Format

For each detected object, the model outputs:
```
Detection = {
    class:       "car"                    (what)
    confidence:  0.92                     (how sure)
    bounding_box: [y_min, x_min,          (where)
                   y_max, x_max]
}
```

### 5.3 Challenges in Object Detection

| Challenge | Description | YOLO's Approach |
|-----------|-------------|----------------|
| Scale variation | Objects can be any size | Multi-scale training, anchors |
| Occlusion | Objects partially hidden | Multiple anchors per cell |
| Viewpoint changes | Different angles | Data augmentation |
| Dense scenes | Many overlapping objects | NMS, high anchor count |
| Small objects | Hard to detect | Passthrough layer, high-res input |
| Speed requirements | Real-time needed | Single-shot architecture |

---

## 6. YOLO — You Only Look Once

### 6.1 The YOLO Revolution

Before YOLO, detection was a multi-step process:
```
R-CNN Pipeline (2014):
  Image → Selective Search (2000 regions) → Warp each → CNN → Classify each
  
  Total computation: 2000 × CNN forward passes = SLOW

YOLO Pipeline (2016):
  Image → Single CNN → All detections at once
  
  Total computation: 1 forward pass = FAST
```

### 6.2 YOLO v1 Core Ideas

**Idea 1: Divide the image into a grid**
```
S × S grid (S = 7 in v1, 13-19 in v2)
Each cell is responsible for detecting objects whose CENTER falls in that cell
```

**Idea 2: Each cell predicts B bounding boxes**
```
Each box: [x, y, w, h, confidence]
- (x, y): Center relative to cell (0-1)
- (w, h): Size relative to image (0-1)
- confidence: P(Object) × IoU(pred, truth)
```

**Idea 3: Each cell predicts C class probabilities**
```
P(Classᵢ | Object) for each class
Final score = P(Object) × IoU × P(Classᵢ | Object)
```

**Idea 4: Single unified loss function**
```
All components (coordinates, confidence, classification) trained together
→ End-to-end differentiable pipeline
```

### 6.3 YOLO v1 Limitations

| Issue | Description |
|-------|-------------|
| Max 2 objects per cell | Struggles with dense scenes |
| Localization errors | Worse than Faster R-CNN for small objects |
| Fixed grid size | Can't adapt to different scales |
| No anchor boxes | Predicts arbitrary shapes (harder to learn) |

---

## 7. YOLO v2 — Better, Faster, Stronger

### 7.1 Key Improvements (used in this project)

#### Improvement 1: Batch Normalization
```
Added after every convolutional layer
Effect: +2% mAP, removed need for dropout
```

#### Improvement 2: High-Resolution Classifier
```
Pre-training steps:
1. Train DarkNet-19 on ImageNet at 224×224 (classification)
2. Fine-tune at 448×448 for 10 epochs
3. Switch to detection mode at 416×416 or 608×608

The high-resolution fine-tuning gives the network time to adjust
to higher-resolution features before attempting detection.
```

#### Improvement 3: Anchor Boxes
```
Instead of predicting arbitrary box shapes:
- Use K-means clustering on training data to find common shapes
- Network predicts OFFSETS from these anchor shapes
- Much easier to learn (smaller prediction space)
```

#### Improvement 4: Direct Location Prediction
```
YOLO v1: Unconstrained (x, y) → box center can be anywhere
YOLO v2: Constrained via sigmoid → center stays in current cell

bₓ = σ(tₓ) + cₓ     (σ output is [0,1] → center in cell)
bᵧ = σ(tᵧ) + cᵧ

This stabilizes training by reducing the prediction space.
```

#### Improvement 5: Passthrough Layer
```
Fine-grained features from earlier layers help detect small objects:

Layer 13 (38×38×512) → 1×1 Conv → (38×38×64) → Reorg → (19×19×256)
                                                            ↓
Layer 20 (19×19×1024) ──────────────────────────→ Concat → (19×19×1280)
```

#### Improvement 6: Multi-Scale Training
```
Every 10 batches, randomly choose a new input size:
{320, 352, 384, 416, 448, 480, 512, 544, 576, 608}

All are multiples of 32 (the network downsampling factor)
→ Network learns to detect at multiple resolutions
→ Same model works for speed (320) or accuracy (608)
```

### 7.2 YOLO v2 Output Tensor

For input 608×608 with 80 COCO classes and 5 anchors:

```
Grid: 19 × 19
Per anchor: 5 (box params) + 80 (classes) = 85
Per cell: 5 anchors × 85 = 425

Output tensor shape: (batch, 19, 19, 425)

Total predictions: 19 × 19 × 5 = 1,805 bounding boxes
```

---

## 8. DarkNet-19 — The Backbone Network

### 8.1 Architecture Design Philosophy

DarkNet-19 was designed specifically for YOLO v2 with these principles:
1. **No fully connected layers** → fully convolutional
2. **1×1 convolutions** → reduce parameters between 3×3 layers
3. **Batch Normalization everywhere** → stable training
4. **Leaky ReLU** → prevent dead neurons

### 8.2 Detailed Layer Table

```
┌─────┬──────────┬─────────┬───────────┬─────────────┬────────────────┐
│ #   │ Type     │ Filters │ Size/Str  │ Output      │ Parameters     │
├─────┼──────────┼─────────┼───────────┼─────────────┼────────────────┤
│  1  │ Conv     │ 32      │ 3×3 / 1   │ 608×608×32  │ 864            │
│  2  │ MaxPool  │ —       │ 2×2 / 2   │ 304×304×32  │ 0              │
│  3  │ Conv     │ 64      │ 3×3 / 1   │ 304×304×64  │ 18,432         │
│  4  │ MaxPool  │ —       │ 2×2 / 2   │ 152×152×64  │ 0              │
│  5  │ Conv     │ 128     │ 3×3 / 1   │ 152×152×128 │ 73,728         │
│  6  │ Conv     │ 64      │ 1×1 / 1   │ 152×152×64  │ 8,192          │
│  7  │ Conv     │ 128     │ 3×3 / 1   │ 152×152×128 │ 73,728         │
│  8  │ MaxPool  │ —       │ 2×2 / 2   │ 76×76×128   │ 0              │
│  9  │ Conv     │ 256     │ 3×3 / 1   │ 76×76×256   │ 294,912        │
│ 10  │ Conv     │ 128     │ 1×1 / 1   │ 76×76×128   │ 32,768         │
│ 11  │ Conv     │ 256     │ 3×3 / 1   │ 76×76×256   │ 294,912        │
│ 12  │ MaxPool  │ —       │ 2×2 / 2   │ 38×38×256   │ 0              │
│ 13  │ Conv     │ 512     │ 3×3 / 1   │ 38×38×512   │ 1,179,648      │
│ 14  │ Conv     │ 256     │ 1×1 / 1   │ 38×38×256   │ 131,072        │
│ 15  │ Conv     │ 512     │ 3×3 / 1   │ 38×38×512   │ 1,179,648      │
│ 16  │ Conv     │ 256     │ 1×1 / 1   │ 38×38×256   │ 131,072        │
│ 17  │ Conv     │ 512     │ 3×3 / 1   │ 38×38×512   │ 1,179,648  ←skip│
│ 18  │ MaxPool  │ —       │ 2×2 / 2   │ 19×19×512   │ 0              │
│ 19  │ Conv     │ 1024    │ 3×3 / 1   │ 19×19×1024  │ 4,718,592      │
│ 20  │ Conv     │ 512     │ 1×1 / 1   │ 19×19×512   │ 524,288        │
│ 21  │ Conv     │ 1024    │ 3×3 / 1   │ 19×19×1024  │ 4,718,592      │
│ 22  │ Conv     │ 512     │ 1×1 / 1   │ 19×19×512   │ 524,288        │
│ 23  │ Conv     │ 1024    │ 3×3 / 1   │ 19×19×1024  │ 4,718,592      │
├─────┼──────────┼─────────┼───────────┼─────────────┼────────────────┤
│     │ TOTAL    │         │           │             │ ~19.8M params  │
└─────┴──────────┴─────────┴───────────┴─────────────┴────────────────┘
```

### 8.3 Bottleneck Block Pattern

```
3×3 Conv (expand)  →  1×1 Conv (compress)  →  3×3 Conv (expand)
    512                     256                     512

This pattern:
1. Processes features at full width (3×3, 512 filters)
2. Compresses to reduce parameters (1×1, 256 filters)
3. Expands back to process at full width (3×3, 512 filters)

Saves parameters: 512×3×3×512 = 2.36M vs 256×1×1×512 + 512×3×3×256 = 131K + 1.18M = 1.31M
→ 44% fewer parameters with similar expressiveness
```

---

## 9. Anchor Boxes & Bounding Box Regression

### 9.1 K-Means for Anchor Generation

```
Training Process:
1. Collect all ground truth bounding boxes from training data
2. Normalize to relative coordinates (w/W, h/H)
3. Run K-means clustering with IoU distance metric:
   d(box, centroid) = 1 - IoU(box, centroid)
4. Cluster centroids become anchor boxes

Result for COCO dataset (k=5):
  Anchor 1: (0.57, 0.68)   — Small  (pedestrians, signs)
  Anchor 2: (1.87, 2.06)   — Medium (sedans, SUVs)
  Anchor 3: (3.34, 5.47)   — Tall   (trucks, buses)
  Anchor 4: (7.88, 3.53)   — Wide   (long vehicles)
  Anchor 5: (9.77, 9.17)   — Large  (close-up vehicles)
```

### 9.2 Box Prediction Mathematics

```
Given: Grid cell (cₓ, cᵧ) and anchor (pᵤ, pₕ)
Network predicts: tₓ, tᵧ, tᵤ, tₕ, tₒ

Decoded bounding box:
  bₓ = σ(tₓ) + cₓ          Center X (constrained to cell)
  bᵧ = σ(tᵧ) + cᵧ          Center Y (constrained to cell)
  bᵤ = pᵤ · exp(tᵤ)        Width (scaled from anchor)
  bₕ = pₕ · exp(tₕ)        Height (scaled from anchor)
  
Objectness confidence:
  Pr(Object) = σ(tₒ)

Class probabilities:
  Pr(Classᵢ|Object) = softmax(t_{c1}, ..., t_{cC})

Final detection score:
  Score = Pr(Object) × max_i[Pr(Classᵢ|Object)]
```

### 9.3 Why Sigmoid for Center Prediction?

```
Without sigmoid: tₓ = 2.5 → center at 2.5 cells away from current cell
  Problem: Network has to learn to predict values for cells far away
  
With sigmoid: σ(tₓ) ∈ (0, 1) → center MUST be within current cell
  Benefit: Each cell only predicts objects centered in that cell
  
This constraint dramatically stabilizes training and improves convergence.
```

### 9.4 Why Exponential for Width/Height?

```
Without exp: Network predicts raw width → can be negative
With exp:    exp(tᵤ) is always positive → valid width

Additionally: exp allows predicting multiplicative changes to anchor size
  tᵤ = 0  → bᵤ = pᵤ × 1 = anchor width (no change)
  tᵤ = 1  → bᵤ = pᵤ × 2.72 (2.72x larger)
  tᵤ = -1 → bᵤ = pᵤ × 0.37 (smaller)
```

---

## 10. Loss Function — Multi-Task Learning

### 10.1 YOLO Loss Components

The YOLO loss simultaneously optimizes four objectives:

```
L = λ_coord · L_coordinate + λ_obj · L_objectness + λ_noobj · L_no_object + λ_class · L_classification

Default weights:
  λ_coord = 1    (coordinate regression)
  λ_obj   = 5    (object confidence — weighted higher because most cells have no objects)
  λ_noobj = 1    (no-object confidence)
  λ_class = 1    (classification)
```

### 10.2 Coordinate Loss (Localization)

```
L_coord = Σ_cells Σ_anchors 𝟙ᵢⱼ^obj [
    (σ(tₓᵢ) - σ(t̂ₓᵢ))²  +  (σ(tᵧᵢ) - σ(t̂ᵧᵢ))²  +
    (tᵤᵢ     - t̂ᵤᵢ)²     +  (tₕᵢ     - t̂ₕᵢ)²
]

Where:
  𝟙ᵢⱼ^obj = 1 if anchor j in cell i is "responsible" for the object
           = 0 otherwise (only penalize the best-matching anchor)
  
  t̂ₓ = ground truth x offset within cell
  t̂ᵤ = log(ground truth width / anchor width)
```

### 10.3 Objectness Loss

```
For cells WITH objects (anchor is responsible):
L_obj = Σ_cells Σ_anchors 𝟙ᵢⱼ^obj · (σ(tₒᵢ) - 1)²
  → Pushes confidence toward 1 for correct detections

For cells WITHOUT objects:
L_noobj = Σ_cells Σ_anchors 𝟙ᵢⱼ^noobj · (σ(tₒᵢ) - 0)²
  → Pushes confidence toward 0 for background
  
Why different weights?
  Most cells (>>90%) contain no objects
  Without weighting, the loss would be dominated by "no object" predictions
  λ_obj = 5 ensures object cells have sufficient gradient signal
```

### 10.4 Classification Loss

```
L_class = Σ_cells Σ_anchors 𝟙ᵢⱼ^obj · Σ_c (p(c) - p̂(c))²

Where:
  p(c) = predicted class probability (from softmax)
  p̂(c) = ground truth (one-hot vector)
  
Only computed for cells/anchors that ARE responsible for a detection
→ Don't penalize classification for background cells
```

### 10.5 Total Loss in Code

```python
# From yad2k/models/keras_yolo.py — simplified
total_loss = 0.5 * (
    coordinates_loss_sum +       # L_coord
    confidence_loss_sum +        # L_obj + L_noobj
    classification_loss_sum      # L_class
)
```

---

## 11. Non-Maximum Suppression (NMS)

### 11.1 The Overlapping Detection Problem

After filtering by confidence, multiple anchors may fire for the same object:

```
For a single car:
  Anchor 1 in cell (5,3): conf=0.92, IoU with truth=0.85
  Anchor 2 in cell (5,4): conf=0.78, IoU with truth=0.60
  Anchor 3 in cell (5,3): conf=0.65, IoU with truth=0.45

All three detect the SAME car → need to keep only the best one
```

### 11.2 NMS Algorithm (Step by Step)

```
Input: N boxes with scores, IoU threshold τ

Step 1: Sort by score: [Box_A (0.92), Box_B (0.78), Box_C (0.65)]

Step 2: Select Box_A (best) → KEEP

Step 3: Compute IoU(Box_A, Box_B) = 0.75 > τ(0.5) → REMOVE Box_B
         Compute IoU(Box_A, Box_C) = 0.60 > τ(0.5) → REMOVE Box_C

Step 4: No boxes remain → DONE

Output: [Box_A]
```

### 11.3 Per-Class NMS

```
This project uses PER-CLASS NMS:
  
  NMS is applied INDEPENDENTLY for each class:
  - All "car" boxes → NMS → surviving car boxes
  - All "truck" boxes → NMS → surviving truck boxes
  - All "person" boxes → NMS → surviving person boxes
  
  A car box can overlap with a person box without suppression!
  This is correct because they are DIFFERENT objects.
```

### 11.4 IoU Threshold Selection

```
Low τ (0.3):  Aggressive suppression → fewer duplicates, may miss nearby objects
Medium τ (0.5): Balanced → standard choice for PASCAL VOC
High τ (0.7):  Lenient suppression → keeps more boxes, risk of duplicates

This project uses τ = 0.5 by default (configurable)
```

---

## 12. Intersection over Union (IoU)

### 12.1 Geometric Definition

```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)
          = Area(Intersection) / (Area(A) + Area(B) - Area(Intersection))

Properties:
  IoU ∈ [0, 1]
  IoU = 0 → no overlap
  IoU = 1 → perfect overlap
  IoU is symmetric: IoU(A,B) = IoU(B,A)
```

### 12.2 Computation

```
Box A: (y1_a, x1_a, y2_a, x2_a) = (top, left, bottom, right)
Box B: (y1_b, x1_b, y2_b, x2_b)

Intersection:
  y1_i = max(y1_a, y1_b)
  x1_i = max(x1_a, x1_b)
  y2_i = min(y2_a, y2_b)
  x2_i = min(x2_a, x2_b)
  
  inter_area = max(0, y2_i - y1_i) × max(0, x2_i - x1_i)

Union:
  area_A = (y2_a - y1_a) × (x2_a - x1_a)
  area_B = (y2_b - y1_b) × (x2_b - x1_b)
  union_area = area_A + area_B - inter_area

IoU = inter_area / union_area
```

### 12.3 IoU in YOLO (Multiple Uses)

| Where | Purpose | Threshold |
|-------|---------|-----------|
| **Training** | Match anchors to GT | Best IoU assignment |
| **Training** | No-object vs object | IoU > 0.6 = object exists |
| **Inference** | NMS suppression | IoU > 0.5 = same object |
| **Evaluation** | TP/FP determination | IoU > 0.5 = correct detection |

### 12.4 Generalized IoU (GIoU) — Advanced

```
GIoU = IoU - (Area(C) - Area(A ∪ B)) / Area(C)

Where C is the smallest convex hull enclosing both A and B.

GIoU ∈ [-1, 1]
GIoU handles non-overlapping boxes (IoU = 0 for all non-overlapping)
Used in more recent YOLO versions (v3+)
```

---

## 13. Evaluation Metrics

### 13.1 Precision & Recall

```
Precision = TP / (TP + FP)
  "Of all detections, what fraction are correct?"
  High precision = few false positives

Recall = TP / (TP + FN)
  "Of all ground truths, what fraction did we detect?"
  High recall = few missed objects

F1 = 2 × (Precision × Recall) / (Precision + Recall)
  Harmonic mean — balances precision and recall
```

### 13.2 Average Precision (AP)

```
For each class:
1. Sort all detections by confidence (descending)
2. For each detection, compute cumulative precision and recall
3. Plot the Precision-Recall curve
4. AP = Area under the P-R curve

Example P-R curve calculation:
  Detection | Correct? | Precision | Recall
  Box1 0.95 |   TP     |  1/1=1.00 | 1/5=0.20
  Box2 0.88 |   TP     |  2/2=1.00 | 2/5=0.40
  Box3 0.75 |   FP     |  2/3=0.67 | 2/5=0.40
  Box4 0.70 |   TP     |  3/4=0.75 | 3/5=0.60
  Box5 0.60 |   FP     |  3/5=0.60 | 3/5=0.60
```

### 13.3 Mean Average Precision (mAP)

```
mAP = (1/C) Σ_c AP(c)

Where C = number of classes with ground truth annotations

PASCAL VOC mAP@0.5:  IoU threshold = 0.5
COCO mAP:           Average over IoU thresholds {0.5, 0.55, ..., 0.95}
                     This is much stricter and gives lower numbers
```

### 13.4 Metric Comparison

| Metric | YOLO v2 Value | What It Tells |
|--------|--------------|---------------|
| mAP@0.5 | 78.6% | General detection accuracy |
| mAP@[.5:.95] | ~44% | Precise localization accuracy |
| FPS | 40 | Real-time capability |
| Recall@IOU0.5 | ~85% | Object coverage |

---

## 14. Transfer Learning & Pre-trained Models

### 14.1 Why Transfer Learning?

```
Training from scratch:         Transfer learning:
  Random weights                Pre-trained weights (ImageNet)
  Needs millions of images      Works with thousands
  Weeks of training             Hours of fine-tuning
  High GPU cost                 Moderate GPU cost
```

### 14.2 YOLO v2 Training Strategy

```
Phase 1: Classification Pre-training
  - DarkNet-19 on ImageNet (1000 classes)
  - 224×224 resolution
  - ~160 epochs
  → Learns general visual features

Phase 2: High-Resolution Fine-tuning
  - Same DarkNet-19 on ImageNet
  - 448×448 resolution
  - 10 epochs
  → Adapts to higher resolution

Phase 3: Detection Training
  - Add YOLO detection layers
  - COCO/VOC dataset
  - 608×608 resolution (multi-scale)
  - ~160 epochs
  → Learns object detection
```

### 14.3 In This Project

```
We use a PRETRAINED model:
  - Already trained through all three phases
  - Saved as TensorFlow SavedModel (model_data/)
  - Can detect 80 COCO classes immediately
  - No training required — inference only
  
This is the most common deployment scenario:
  "Download a pre-trained model and use it for detection"
```

---

## 15. Autonomous Driving Context

### 15.1 Perception Pipeline

```
Camera → Object Detection → Tracking → Prediction → Planning → Control
         (This Project)     ↓
                    ┌────────────────┐
                    │ Sensor Fusion  │ ← LIDAR, Radar
                    └────────────────┘

Our project handles the DETECTION component:
  Input: Single RGB image from front-facing camera
  Output: List of detected objects with class, location, confidence
```

### 15.2 Critical Classes for Driving

| Class | Importance | Why |
|-------|-----------|-----|
| **Car** | 🔴 Critical | Collision avoidance |
| **Truck** | 🔴 Critical | Large vehicles, blind spots |
| **Pedestrian** | 🔴 Critical | Vulnerable road users |
| **Bicycle** | 🔴 Critical | Vulnerable road users |
| **Traffic Light** | 🟡 Important | Rule compliance |
| **Stop Sign** | 🟡 Important | Rule compliance |
| **Bus** | 🟡 Important | Public transport, large vehicle |
| **Motorbike** | 🟡 Important | Small, fast-moving |

### 15.3 Safety Requirements

```
Autonomous Driving Detection Requirements:
  
  Recall:      >99% for pedestrians (can't miss people)
  Precision:   >95% for all classes (minimize false braking)
  Latency:     <100ms end-to-end (real-time reaction)
  Redundancy:  Multiple sensor + multiple algorithm agreement
  Range:       200m detection for highway, 50m for urban
```

### 15.4 Limitations of Camera-Only Detection

```
Strengths:
  ✅ Rich semantic information (text, colors, signs)
  ✅ High resolution
  ✅ Cheap sensors
  ✅ Good for classification

Limitations:
  ❌ No reliable depth information
  ❌ Poor in darkness/glare/rain
  ❌ Limited by occlusion
  ❌ 2D detection only (no 3D pose)
  
Production systems combine cameras with LIDAR and radar
```

---

## 16. Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| x | Input features / image pixels |
| W, w | Weight matrices / vectors |
| b | Bias vector |
| z | Pre-activation (linear combination) |
| a | Post-activation output |
| σ | Sigmoid function |
| α | Learning rate |
| L | Loss function value |
| ∇ | Gradient operator |
| θ | Model parameters (all weights and biases) |
| ŷ | Model prediction |
| y | Ground truth |
| B | Number of anchor boxes |
| C | Number of classes |
| S | Grid size |
| 𝟙 | Indicator function (1 if true, 0 if false) |
| IoU | Intersection over Union |
| AP | Average Precision |
| mAP | Mean Average Precision |
| TP | True Positive |
| FP | False Positive |
| FN | False Negative |
| NMS | Non-Maximum Suppression |
| BN | Batch Normalization |
| (tₓ, tᵧ, tᵤ, tₕ, tₒ) | Raw box predictions |
| (bₓ, bᵧ, bᵤ, bₕ) | Decoded box coordinates |
| (cₓ, cᵧ) | Grid cell offset |
| (pᵤ, pₕ) | Anchor box dimensions |

---

## 17. Glossary

| Term | Definition |
|------|-----------|
| **Anchor Box** | Pre-defined box shape used as a reference for prediction |
| **Backbone** | Feature extraction network (DarkNet-19 in YOLO v2) |
| **Batch Normalization** | Technique to normalize layer inputs for stable training |
| **Bounding Box** | Rectangular region marking detected object location |
| **COCO** | Common Objects in Context — 80-class detection dataset |
| **Confidence Score** | Probability that a detection is correct |
| **Convolution** | Mathematical operation that applies a filter to extract features |
| **DarkNet** | Custom neural network architecture designed for YOLO |
| **Detection Head** | Final layers that produce box predictions and class scores |
| **Epoch** | One complete pass through the training dataset |
| **Feature Map** | Output of a convolutional layer — represents learned features |
| **Grid Cell** | One subdivision of the YOLO prediction grid |
| **Ground Truth** | Human-annotated correct detection for evaluation |
| **Inference** | Running a trained model on new data to make predictions |
| **IoU** | Intersection over Union — overlap measure between two boxes |
| **Kernel/Filter** | Small matrix of learnable weights in convolution |
| **Leaky ReLU** | Activation allowing small negative gradients |
| **mAP** | Mean Average Precision — standard detection accuracy metric |
| **NMS** | Non-Maximum Suppression — removes duplicate detections |
| **Objectness** | Predicted probability that a box contains any object |
| **PASCAL VOC** | Visual Object Classes — 20-class detection dataset |
| **Passthrough Layer** | Skip connection bringing fine-grained features to detection |
| **Receptive Field** | Input region that influences a single output neuron |
| **SavedModel** | TensorFlow's standard model serialization format |
| **Softmax** | Function converting logits to probability distribution |
| **Stride** | Step size when sliding a convolution kernel |
| **Transfer Learning** | Reusing pre-trained model weights for a new task |
| **YOLO** | You Only Look Once — single-shot object detection architecture |

---

*This document serves as a complete theoretical reference for the Autonomous Driving Car Detection project.
Every concept mentioned in the code and README is explained here with mathematical foundations.*
