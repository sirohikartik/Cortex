# 🧠 Cortex

> Clean, educational, from-scratch implementations of fundamental Machine Learning and Deep Learning algorithms in PyTorch and NumPy.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📌 Overview

**Cortex** is a collection of core Machine Learning (ML) and Deep Learning (DL) algorithms implemented from first principles. Rather than relying on high-level abstractions like scikit-learn or pre-built Transformer libraries, every mathematical operation—from gradient descent weight updates and KD-Tree partitioning to scaled dot-product attention and causal masking—is implemented directly using tensor operations.

The goal of this repository is deep mechanical sympathy: understanding how ML/DL models work under the hood by implementing the mathematics step by step.

---

## 📁 Repository Structure

```text
Cortex/
├── ML/
│   ├── LinearRegression/
│   │   ├── _gd.py              # Batch, Mini-Batch, & Stochastic Gradient Descent
│   │   └── _normal.py          # Normal Equation with Ridge (L2) Regularization
│   ├── LogisticRegression/
│   │   └── _gd.py              # Sigmoid activation & Batch Gradient Descent
│   ├── KNN/
│   │   ├── _knn.py             # Naive Euclidean distance Classifier & Regressor
│   │   └── _kdtree.py          # KD-Tree spatial indexing & nearest-neighbor search
│   ├── NaiveBayes/
│   │   └── _naive_bayes.py     # Gaussian Naive Bayes with Log-Likelihood
│   └── DecisionTree/
│       └── _dt.py              # Decision Tree using Entropy & Information Gain
│
├── DL/
│   └── Transformers/
│       ├── _self_attn.py       # Scaled Dot-Product Self-Attention with test harness
│       ├── _multi_head_attn.py # Multi-Head Attention (MHA) module
│       ├── __pos_encoding.py   # Sinusoidal Positional Encoding (naive & vectorized)
│       ├── _casual_mask.py     # Causal (Autoregressive) Masking
│       ├── _encoder.py         # Complete Transformer Encoder Block (LayerNorm + FFN)
│       └── _model.py           # Input Embedding & Transformer architectural components
│
├── .gitignore
└── README.md
```

---

## 🔬 Implemented Algorithms

### 1. Classical Machine Learning (`/ML`)

* **Linear Regression**
  * **Gradient Descent Variations (`_gd.py`)**: Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent (MBGD) with vectorized weight updates.
  * **Closed-Form Normal Equation (`_normal.py`)**: Analytical solution via $\theta = (X^T X + \lambda I)^{-1} X^T y$ incorporating $L_2$ Ridge Regularization.
* **Logistic Regression (`_gd.py`)**
  * Binary classification using the Sigmoid link function $\sigma(z) = \frac{1}{1 + e^{-z}}$ and cross-entropy gradient descent optimization.
* **K-Nearest Neighbors (KNN)**
  * **Brute-Force / Euclidean (`_knn.py`)**: Vectorized distance calculation with top-$k$ voting for classification and mean pooling for regression.
  * **KD-Tree (`_kdtree.py`)**: Multi-dimensional binary search tree implementation for efficient spatial queries.
* **Naive Bayes (`_naive_bayes.py`)**
  * Gaussian Naive Bayes using feature-wise mean and variance calculations with numerically stable log-likelihood summation.
* **Decision Trees (`_dt.py`)**
  * Binary splitting using Shannon Entropy $H(S) = -\sum p_i \log_2(p_i)$ and Information Gain optimization.

---

### 2. Deep Learning & Transformers (`/DL`)

* **Attention Mechanisms**
  * **Scaled Dot-Product Attention (`_self_attn.py`)**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$.
  * **Multi-Head Attention (`_multi_head_attn.py`)**: Linear query, key, value projection across parallel attention heads with tensor concatenation.
* **Positional Encodings (`__pos_encoding.py`)**
  * Sine/Cosine frequency positional encodings for sequence awareness, implemented in both loop-based and vectorized tensor formats:
    $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
* **Causal Masking (`_casual_mask.py`)**
  * Autoregressive lower-triangular mask (`torch.tril`) setting future token attention weights to $-\infty$.
* **Transformer Building Blocks (`_encoder.py`, `_model.py`)**
  * Complete Transformer Encoder Block featuring Multi-Head Attention, residual skip connections, Layer Normalization (`nn.LayerNorm`), and Feed-Forward Networks (FFN).
  * Scaled `InputEmbedding` module with $\sqrt{d_{model}}$ normalization.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* PyTorch (CPU or GPU/MPS)
* NumPy

### Installation

```bash
# Clone the repository
git clone https://github.com/sirohikartik/Cortex.git
cd Cortex

# Create and activate a virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install torch numpy
```

### Running Examples

Run any standalone script or its built-in verification suite:

```bash
# Run Linear Regression Gradient Descent tests
python3 ML/LinearRegression/_gd.py

# Run Causal Mask tests
python3 DL/Transformers/_casual_mask.py

# Run Self-Attention test harness
python3 DL/Transformers/_self_attn.py
```

---

## 🛠️ Design Philosophy & Transparency

* **From First Principles**: Avoid high-level abstractions to clarify internal tensor transformations and mathematical derivation.
* **Vectorized & PyTorch-Native**: Leverage PyTorch tensor operations for fast matrix arithmetic while preserving clarity.
* **Implementation Authenticity**: All core algorithmic logic and mathematical computations are written by hand. Synthetic testing harnesses and verification boilerplate may leverage AI assistance.

---

## 🗺️ Roadmap

- [ ] Complete Transformer Decoder Block
- [ ] Full Seq2Seq Transformer translation pipeline
- [ ] Support Vector Machines (SVM with SMO algorithm)
- [ ] Principal Component Analysis (PCA via SVD / Eigendecomposition)
- [ ] Random Forests with bagging and feature subsampling

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
