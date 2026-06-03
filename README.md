Attention-Driven Bi-LSTM Networks for Robust Twitter Sentiment Analysis

This repository contains a robust, end-to-end deep learning pipeline for Twitter sentiment analysis. By combining Bidirectional Long Short-Term Memory (Bi-LSTM) networks with a custom Contextual Attention Mechanism, the model successfully captures long-range dependencies and highlights the most emotionally weighted words in a tweet. Pre-trained GloVe (Global Vectors for Word Representation) embeddings are utilized to ground the network with rich semantic prior knowledge.

🚀 Features

Advanced Text Preprocessing: Handles contractions, converts emojis to text descriptions, retains critical negations (not, no, nor), and strips out platform-specific noise (URLs, user handles, punctuation).

Pre-trained Semantic Layer: Leverages the 100-dimensional Stanford GloVe embeddings to handle out-of-vocabulary contexts gracefully.

Custom Attention Layer: Implements a trainable single-hop attention mechanism to weigh word importance dynamically across hidden states.

Seamless Kaggle Integration: Uses kagglehub for fast, automated, and secure programmatic dataset downloads.

🧠 Model Architecture

The network processes text sequentially through a multi-tiered deep learning architecture:

Input Sequences (Max Len: 100) ->
       
[GloVe Embedding Layer] (100d, Frozen)->

[Bidirectional LSTM] (128 Units, Return Sequences)->
       
  [Attention Layer] ───► Calculates alpha weights dynamically->
       
   [Dropout] (0.3)->
       
 [Dense Layer] (64 Units, ReLU)->
       
 [Output Layer] (1 Unit, Sigmoid for Binary Classification)

 The Attention Mechanism:
 <img width="810" height="247" alt="image" src="https://github.com/user-attachments/assets/d577dcb2-8501-4022-8e4f-69cc034945ae" />

 📊 Dataset
 
The model is trained on the Sentiment140 dataset, which contains 1.6 million tweets automatically annotated for positive and negative sentiment.

Target Labels: Original labels 0 (Negative) and 4 (Positive) are structurally re-mapped to standard binary classification inputs 0 and 1.

📝 Configuration Parameters

MAX_WORDS: 50,000 (Vocabulary limit)

MAX_LEN: 100 (Sequence padding limit)

EMBEDDING_DIM: 100 (GloVe embedding vector size)

LSTM Units: 128 (Bi-LSTM output size: 256 dimensions)

Batch Size: 256

Epochs: 1 (Adjust upwards for full model convergence) 
