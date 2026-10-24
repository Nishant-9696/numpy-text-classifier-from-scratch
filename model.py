"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
import re

def clean_text(text: str) -> str:
    text = text.lower()
    result = re.sub(r'[^a-zA-Z]', ' ', text)
    return result.strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    text=text.split()
    return text

# Step 3 - tokenize_corpus
import re
def tokenize_corpus(texts: list) -> list:
    result=[]
    for i in texts:
        text= i[0] if isinstance(i,list) else i
        text =text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ',text)
        text =text.split()
        result.append(text)
    return result

# Step 4 - split_train_val_test_indices
import numpy as np
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    np.random.seed(seed)
    indices=np.arange(n_samples)
    np.random.shuffle(indices)
    n_val=int(n_samples*val_fraction)
    n_test=int(n_samples*test_fraction)
    n_train=n_samples-n_val-n_test
    train=indices[:n_train]
    val=indices[n_train:n_train+n_val]
    test=indices[n_train+n_val:]
    return train,val,test

# Step 5 - count_word_frequencies
from collections import Counter
def count_word_frequencies(tokenized_docs: list) -> dict:
    words=[]
    if len(tokenized_docs) == 0:
        return {}
    for w in tokenized_docs:
        words.extend(w)
    return Counter(words)

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    top_words = sorted_words[:max_size]

    vocabulary = {
        word: index
        for index, (word, count) in enumerate(top_words)
    }

    return vocabulary

# Step 7 - tokens_to_bow
import numpy as np

def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    bow = np.zeros(len(vocab), dtype=float)

    for word in tokens:
        if word in vocab:
            bow[vocab[word]] += 1

    return bow

# Step 8 - corpus_to_bow_matrix
import numpy as np

def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    V = len(vocab)

    if len(tokenized_docs) == 0:
        return np.zeros((0, V), dtype=float)

    matrix = []

    for tokens in tokenized_docs:
        vector = np.zeros(V, dtype=float)

        for word in tokens:
            if word in vocab:
                vector[vocab[word]] += 1

        matrix.append(vector)

    return np.array(matrix, dtype=float)

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    df = np.sum(bow_matrix > 0, axis=0)
    return df

# Step 10 - compute_idf
import numpy as np

def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # Compute smoothed IDF
    idf = np.log((n_docs + 1) / (df + 1)) + 1
    return idf

# Step 11 - transform_tfidf
import numpy as np

def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # Multiply each term count by its corresponding IDF weight
    return bow_matrix * idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    df = compute_document_frequencies(bow_train)
    idf = compute_idf(df, bow_train.shape[0])
    return idf

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)

    pos = z >= 0
    result[pos] = 1 / (1 + np.exp(-z[pos]))

    exp_z = np.exp(z[~pos])
    result[~pos] = exp_z / (1 + exp_z)

    return result

# Step 14 - logistic_predict_proba
import math
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    r=[]
    for i in range(len(X)):
        z=w @ X[i]+b
        s=1/(1+math.exp(-z))
        r.append(s)
    return r

# Step 15 - binary_cross_entropy
import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

    # Compute binary cross-entropy
    bce = -np.mean(
        y_true * np.log(y_prob) +
        (1 - y_true) * np.log(1 - y_prob)
    )

    # Compute L2 regularization penalty
    l2_penalty = l2_lambda * np.sum(w ** 2) / 2

    # Return total loss
    return float(bce + l2_penalty)

# Step 16 - logistic_gradients (not yet solved)
# TODO: implement

# Step 17 - initialize_logistic_params (not yet solved)
# TODO: implement

# Step 18 - gradient_descent_step (not yet solved)
# TODO: implement

# Step 19 - train_logistic_regression (not yet solved)
# TODO: implement

# Step 20 - predict_labels (not yet solved)
# TODO: implement

# Step 21 - confusion_counts (not yet solved)
# TODO: implement

# Step 22 - metrics_from_counts (not yet solved)
# TODO: implement

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

