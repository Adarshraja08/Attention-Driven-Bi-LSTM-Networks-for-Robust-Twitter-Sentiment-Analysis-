# Transformer-Inspired Attention-Based Sentiment Analysis of Twitter Data

## Overview

This project presents an advanced deep learning framework for Twitter sentiment analysis using Bidirectional Long Short-Term Memory (Bi-LSTM) networks combined with an Attention Mechanism and pre-trained GloVe word embeddings.

Traditional sentiment analysis models often struggle to identify the most influential words within a tweet. To address this challenge, the proposed architecture incorporates an attention layer that enables the model to focus on sentiment-bearing words while reducing the influence of less important terms.

The model is trained and evaluated on the Sentiment140 dataset and classifies tweets into positive and negative sentiment categories.

---

# Key Features

* Advanced tweet preprocessing
* Pre-trained GloVe embeddings
* Bidirectional LSTM architecture
* Attention mechanism
* Deep neural network classifier
* Dropout regularization
* Binary sentiment classification
* Confusion matrix visualization
* Performance evaluation using multiple metrics

---

# Dataset

## Sentiment140 Dataset

The project uses the Sentiment140 dataset containing 1.6 million Twitter posts.

### Dataset Statistics

| Attribute       | Value     |
| --------------- | --------- |
| Total Tweets    | 1,600,000 |
| Positive Tweets | 800,000   |
| Negative Tweets | 800,000   |
| Classes         | Binary    |

### Label Mapping

| Original Label | Converted Label |
| -------------- | --------------- |
| 0              | Negative        |
| 4              | Positive (1)    |

Dataset Source:

https://www.kaggle.com/datasets/kazanova/sentiment140

---

# Project Workflow

## Step 1: Data Acquisition

The dataset is downloaded directly from Kaggle.

```python
kagglehub.dataset_download("kazanova/sentiment140")
```

---

## Step 2: Text Preprocessing

Raw tweets undergo extensive cleaning before training.

### Operations Performed

### Lowercasing

```text
I LOVE AI
↓
i love ai
```

### Contraction Expansion

```text
can't
↓
cannot
```

### URL Removal

```text
https://example.com
↓
removed
```

### Mention Removal

```text
@username
↓
removed
```

### Emoji Removal

```text
😊
↓
removed
```

### Punctuation Removal

```text
Amazing!!!
↓
Amazing
```

### Number Removal

```text
I bought 2 phones
↓
I bought phones
```

### Stop-word Removal

```text
This is a good movie
↓
good movie
```

---

## Step 3: Data Splitting

The dataset is divided into:

* Training Set: 80%
* Testing Set: 20%

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

## Step 4: Tokenization

Text is converted into integer sequences.

```python
Tokenizer(num_words=367005)
```

The tokenizer creates a vocabulary dictionary mapping words to unique integer IDs.

---

## Step 5: Sequence Padding

All sequences are padded to a fixed length.

```python
pad_sequences(
    maxlen=100
)
```

Example:

```text
[12,45,31]
↓
[0,0,0,...,12,45,31]
```

---

## Step 6: GloVe Embedding Integration

The project utilizes pre-trained GloVe embeddings.

### GloVe Model

```text
glove.6B.100d
```

### Embedding Dimension

```text
100
```

Benefits:

* Captures semantic relationships
* Provides richer word representations
* Improves generalization
* Reduces training time

Example:

```text
king ≈ queen
happy ≈ joyful
```

The embedding matrix is initialized using pre-trained GloVe vectors.

---

# Deep Learning Architecture

## Model Structure

```text
Input Layer
      │
      ▼
Embedding Layer
(GloVe 100D)
      │
      ▼
Bidirectional LSTM
(128 Units)
      │
      ▼
Attention Layer
      │
      ▼
Dropout
      │
      ▼
Dense Layer (64)
      │
      ▼
Output Layer
(Sigmoid)
```

---

## Bidirectional LSTM Layer

The Bi-LSTM processes tweets in both forward and backward directions.

Advantages:

* Captures past context
* Captures future context
* Improves understanding of sentence meaning
* Handles long-range dependencies

Example:

```text
I do not like this movie
```

The word "not" changes the sentiment and Bi-LSTM effectively captures this dependency.

---

## Attention Mechanism

The attention layer is the most important component of the model.

Instead of treating all words equally, attention assigns higher weights to sentiment-relevant words.

Example:

```text
The movie was absolutely fantastic
```

Attention focuses more on:

```text
fantastic
```

and less on:

```text
The
was
absolutely
```

Benefits:

* Improved interpretability
* Better feature extraction
* Enhanced sentiment detection
* Improved classification performance

---

## Dropout Layer

Dropout helps reduce overfitting by randomly deactivating neurons during training.

Benefits:

* Better generalization
* Reduced model variance
* Improved robustness

---

## Dense Neural Network

The extracted attention-weighted features are passed through dense layers.

### Hidden Layer

```text
64 Neurons
ReLU Activation
```

### Output Layer

```text
1 Neuron
Sigmoid Activation
```

Output:

```text
0 → Negative
1 → Positive
```

---

# Model Training

### Loss Function

```text
Binary Crossentropy
```

### Optimizer

```text
Adam
```

### Evaluation Metric

```text
Accuracy
```

---

# Performance Evaluation

The model is evaluated using:

## Accuracy

Measures overall prediction correctness.

## Precision

Measures quality of positive predictions.

## Recall

Measures ability to identify positive samples.

## F1-Score

Balances precision and recall.

## Confusion Matrix

Visualizes:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

# Technologies Used

## Programming Language

* Python

## Deep Learning

* TensorFlow
* Keras

## NLP

* NLTK
* Emoji
* Contractions

## Word Embeddings

* GloVe 100D

## Machine Learning

* Scikit-Learn

## Data Processing

* NumPy
* Pandas

## Visualization

* Matplotlib
* Seaborn

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/twitter-attention-sentiment-analysis.git
```

Install dependencies:

```bash
pip install tensorflow
pip install scikit-learn
pip install nltk
pip install emoji
pip install contractions
pip install kagglehub
```

---

# Running the Project

Open the notebook:

```bash
jupyter notebook TSA_TRANSF.ipynb
```

Run all cells sequentially.

---

# Results

The proposed model successfully combines:

* Semantic knowledge from GloVe embeddings
* Context learning through Bi-LSTM
* Word importance identification using Attention

This architecture achieves highly effective sentiment classification for Twitter data while providing better contextual understanding than conventional deep learning models.

---

# Future Improvements

* Replace Bi-LSTM with Transformer Encoder
* Integrate BERT embeddings
* Multi-class sentiment classification
* Real-time Twitter stream analysis
* Explainable AI visualization
* Deployment using Flask or FastAPI

---

# Author

Adarsh Raja

M.Tech CSE
NIT KURUKSHETRA

Research Interests:
Natural Language Processing, Deep Learning, Artificial Intelligence, and Sentiment Analysis
