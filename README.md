# Word Vectorization Experiment for Text Classification

## 📌 Assignment Information
- **Subject:** Machine Learning in CYS (24CYS214)
- **Deadline:** 30/03/2026
- **Dataset:** 20 Newsgroups (18,846 documents)

## 🎯 Objective
Compare three word vectorization methods (Bag of Words, TF-IDF, Hashing Vectorizer) for text classification, analyzing the trade-off between performance and time complexity.

## 🔧 Implementation Details

### Preprocessing Steps
- Lowercasing
- Special character removal
- Stopword removal (NLTK)
- Porter Stemming
- N-gram generation (unigrams + bigrams)

### Vectorization Methods
| Method | Parameters |
|--------|------------|
| Bag of Words (BoW) | max_features=8000, ngram_range=(1,2) |
| TF-IDF | max_features=8000, ngram_range=(1,2) |
| Hashing Vectorizer | n_features=8000, alternate_sign=False |

### Classifier
- **Algorithm:** Logistic Regression
- **Regularization:** C=1.0 (L2 regularization)
- **Train/Test Split:** 80/20

## 📊 Experimental Results

### Performance Metrics

| Method | Accuracy | Precision | Recall | F1-Score | Total Time (s) |
|--------|----------|-----------|--------|----------|----------------|
| Bag of Words (BoW) | 0.8631 | 0.8507 | 0.8955 | 0.8725 | 8.56 |
| TF-IDF | **0.8721** | **0.8551** | **0.9097** | **0.8816** | 7.40 |
| Hashing Vectorizer | 0.8523 | 0.8371 | 0.8910 | 0.8632 | **2.59** |

### Confusion Matrices

#### Bag of Words
