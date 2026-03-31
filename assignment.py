# ============================================
# ML Assignment: Word Vectorization Experiment
# Subject: Machine Learning in CYS (24CYS214)
# Dataset: 20 Newsgroups (FULL - 18,846 samples)
# ============================================

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ============================================
# 1. LOAD FULL 20 NEWSGROUPS DATASET
# ============================================

print("=" * 60)
print("LOADING FULL 20 NEWSGROUPS DATASET (18,846 samples)")
print("=" * 60)

# Split into two groups for binary classification
group1_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                      'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos', 'rec.motorcycles',
                      'rec.sport.baseball', 'rec.sport.hockey', 'sci.space']
                      
group2_categories = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc',
                      'sci.med', 'sci.crypt', 'sci.electronics', 'misc.forsale',
                      'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc']

print(f"Group 1 (Technology/Sports/Space): {len(group1_categories)} categories")
print(f"Group 2 (Religion/Medicine/Politics): {len(group2_categories)} categories")

# Load the FULL dataset
print("\nLoading Group 1 (Technology/Sports/Space)...")
newsgroups1 = fetch_20newsgroups(
    subset='all', 
    categories=group1_categories,
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=42
)

print("Loading Group 2 (Religion/Medicine/Politics)...")
newsgroups2 = fetch_20newsgroups(
    subset='all', 
    categories=group2_categories,
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=42
)

# Combine datasets
data1 = pd.DataFrame({
    'review': newsgroups1.data,
    'label': 'group1'
})

data2 = pd.DataFrame({
    'review': newsgroups2.data,
    'label': 'group2'
})

data = pd.concat([data1, data2], ignore_index=True)

print(f"\n✅ Dataset loaded successfully!")
print(f"Total samples: {len(data)}")
print(f"\nClass distribution:")
print(data['label'].value_counts())
print(f"\nGroup 1 samples: {len(data[data['label']=='group1'])}")
print(f"Group 2 samples: {len(data[data['label']=='group2'])}")

# ============================================
# 2. TEXT PREPROCESSING
# ============================================

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization and stopword removal + stemming
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 2]
    
    # Join back
    return ' '.join(words)

print("\n" + "=" * 60)
print("PREPROCESSING TEXT DATA...")
print("=" * 60)
print(f"Processing {len(data)} documents (this may take 2-3 minutes)...")

start_preprocess = time.time()
data['cleaned_review'] = data['review'].apply(preprocess_text)
preprocess_time = time.time() - start_preprocess

print(f"\n✅ Preprocessing completed in {preprocess_time:.2f} seconds!")

print(f"\nSample original review:")
print(data['review'].iloc[0][:300])
print(f"\nSample preprocessed review:")
print(data['cleaned_review'].iloc[0][:300])

# ============================================
# 3. DATA EXPLORATION
# ============================================

print("\n" + "=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

# Review length analysis
data['review_length'] = data['review'].apply(len)
data['cleaned_words'] = data['cleaned_review'].apply(lambda x: len(x.split()))

print(f"\nReview length statistics:")
print(f"Original reviews - Mean length: {data['review_length'].mean():.1f} chars")
print(f"Original reviews - Min length: {data['review_length'].min():.1f} chars")
print(f"Original reviews - Max length: {data['review_length'].max():.1f} chars")
print(f"Cleaned reviews - Mean words: {data['cleaned_words'].mean():.1f}")

# Most common words by category
from collections import Counter

group1_reviews = data[data['label'] == 'group1']['cleaned_review']
group2_reviews = data[data['label'] == 'group2']['cleaned_review']

group1_words = ' '.join(group1_reviews).split()
group2_words = ' '.join(group2_reviews).split()

print(f"\nTop 15 words in GROUP 1 (Technology/Sports/Space):")
for word, count in Counter(group1_words).most_common(15):
    print(f"   {word}: {count}")

print(f"\nTop 15 words in GROUP 2 (Religion/Medicine/Politics):")
for word, count in Counter(group2_words).most_common(15):
    print(f"   {word}: {count}")

# Find overlapping words
group1_set = set([w for w, _ in Counter(group1_words).most_common(300)])
group2_set = set([w for w, _ in Counter(group2_words).most_common(300)])
overlap = group1_set.intersection(group2_set)
print(f"\nNumber of overlapping words in top 300: {len(overlap)}")
print(f"Examples of overlapping words: {list(overlap)[:20]}")

# ============================================
# 4. PREPARE DATA FOR MODELING
# ============================================

# Prepare data
X = data['cleaned_review']
y = (data['label'] == 'group1').astype(int)  # 1 for group1, 0 for group2

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size: {len(X_test)}")
print(f"Group1 class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

# ============================================
# 5. VECTORIZATION METHODS AND EVALUATION
# ============================================

# Define vectorizers with larger feature space for full dataset
vectorizers = {
    'Bag of Words (BoW)': CountVectorizer(max_features=8000, min_df=2, ngram_range=(1,2)),
    'TF-IDF': TfidfVectorizer(max_features=8000, min_df=2, ngram_range=(1,2)),
    'Hashing Vectorizer': HashingVectorizer(n_features=8000, alternate_sign=False)
}

# Store results
results = {}
vectorization_times = {}
vocabulary_sizes = {}

print("\n" + "=" * 60)
print("EVALUATING VECTORIZATION METHODS ON FULL DATASET")
print("=" * 60)
print("This may take 3-5 minutes. Please wait...")

for name, vectorizer in vectorizers.items():
    print(f"\n{'='*50}")
    print(f"METHOD: {name}")
    print(f"{'='*50}")
    
    # Measure vectorization time
    start_time = time.time()
    
    # Fit and transform
    print("   Vectorizing text...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    vec_time = time.time() - start_time
    vectorization_times[name] = vec_time
    
    # Get vocabulary size
    vocab_size = X_train_vec.shape[1]
    vocabulary_sizes[name] = vocab_size
    
    # Calculate sparsity
    sparsity = 100 * (1 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]))
    
    print(f"✓ Vectorization time: {vec_time:.2f} seconds")
    print(f"✓ Feature size: {vocab_size:,}")
    print(f"✓ Training data shape: {X_train_vec.shape}")
    print(f"✓ Test data shape: {X_test_vec.shape}")
    print(f"✓ Sparsity: {sparsity:.2f}%")
    
    # Train classifier
    print("   Training classifier...")
    classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0, n_jobs=-1)
    
    # Measure training time
    train_start = time.time()
    classifier.fit(X_train_vec, y_train)
    train_time = time.time() - train_start
    
    # Predictions
    print("   Making predictions...")
    y_pred = classifier.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation (3-fold on sample for speed)
    print("   Performing cross-validation...")
    cv_scores = cross_val_score(classifier, X_train_vec[:5000], y_train[:5000], cv=3, scoring='f1', n_jobs=-1)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean F1': cv_scores.mean(),
        'CV Std F1': cv_scores.std(),
        'Training Time': train_time,
        'Vectorization Time': vec_time,
        'Total Time': vec_time + train_time,
        'Vocabulary Size': vocab_size,
        'Sparsity': sparsity
    }
    
    # Print results
    print(f"\n📊 Classification Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n⏱️ Time Analysis:")
    print(f"   Vectorization: {vec_time:.2f}s")
    print(f"   Training:      {train_time:.2f}s")
    print(f"   Total:         {vec_time + train_time:.2f}s")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Negatives:  {cm[0,0]:5,d} | False Positives: {cm[0,1]:5,d}")
    print(f"   False Negatives: {cm[1,0]:5,d} | True Positives:  {cm[1,1]:5,d}")

# ============================================
# 6. COMPARATIVE ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("COMPARATIVE ANALYSIS - RESULTS SUMMARY")
print("=" * 60)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(4)

print("\n📊 PERFORMANCE COMPARISON TABLE:")
print(comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Total Time', 'Vocabulary Size']].to_string())

# ============================================
# 7. VISUALIZATIONS
# ============================================

print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS...")
print("=" * 60)

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Performance Metrics Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results))
width = 0.2

for i, metric in enumerate(metrics):
    values = [results[method][metric] for method in results.keys()]
    axes[0, 0].bar(x + i*width, values, width, label=metric)

axes[0, 0].set_xlabel('Vectorization Methods', fontsize=12)
axes[0, 0].set_ylabel('Score', fontsize=12)
axes[0, 0].set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x + width * 1.5)
axes[0, 0].set_xticklabels(results.keys(), rotation=45, ha='right')
axes[0, 0].legend(loc='lower right')
axes[0, 0].set_ylim([0.7, 1.0])
axes[0, 0].grid(True, alpha=0.3)

# 2. Time Comparison
methods = list(results.keys())
vec_times = [vectorization_times[m] for m in methods]
train_times = [results[m]['Training Time'] for m in methods]

x_pos = np.arange(len(methods))
axes[0, 1].bar(x_pos - 0.2, vec_times, 0.4, label='Vectorization Time', color='#3498db')
axes[0, 1].bar(x_pos + 0.2, train_times, 0.4, label='Training Time', color='#e74c3c')
axes[0, 1].set_xlabel('Vectorization Methods', fontsize=12)
axes[0, 1].set_ylabel('Time (seconds)', fontsize=12)
axes[0, 1].set_title('Time Complexity Analysis', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. F1-Score vs Time Trade-off
f1_scores = [results[m]['F1-Score'] for m in methods]
total_times = [results[m]['Total Time'] for m in methods]

colors = ['#2ecc71', '#3498db', '#e74c3c']
axes[1, 0].scatter(total_times, f1_scores, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)
for i, method in enumerate(methods):
    axes[1, 0].annotate(method, (total_times[i], f1_scores[i]), xytext=(10, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
axes[1, 0].set_xlabel('Total Time (seconds)', fontsize=12)
axes[1, 0].set_ylabel('F1-Score', fontsize=12)
axes[1, 0].set_title('Performance vs Time Complexity Trade-off', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature Size Comparison
vocab_sizes = [vocabulary_sizes[m] for m in methods]
bars = axes[1, 1].bar(methods, vocab_sizes, color=colors, edgecolor='black', linewidth=1)
axes[1, 1].set_xlabel('Vectorization Methods', fontsize=12)
axes[1, 1].set_ylabel('Feature Size', fontsize=12)
axes[1, 1].set_title('Feature Space Comparison', fontsize=14, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, vocab_sizes):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                    f'{val:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('vectorization_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Visualization saved as 'vectorization_analysis.png'")

# ============================================
# 8. DETAILED CLASSIFICATION REPORTS
# ============================================

print("\n" + "=" * 60)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 60)

for name, vectorizer in vectorizers.items():
    print(f"\n{name}:")
    print("-" * 50)
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    classifier = LogisticRegression(max_iter=1000, random_state=42, C=1.0, n_jobs=-1)
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    
    print(classification_report(y_test, y_pred, target_names=['Group2', 'Group1']))

# ============================================
# 9. EXPORT RESULTS
# ============================================

comparison_df.to_csv('results_comparison.csv')
print("\n✅ Results saved to 'results_comparison.csv'")

# Print summary
print("\n" + "=" * 60)
print("SUMMARY FOR REPORT")
print("=" * 60)

print("\n📊 Best Performing Method (by F1-Score):")
best_method = max(results.keys(), key=lambda x: results[x]['F1-Score'])
print(f"   → {best_method} with F1-Score: {results[best_method]['F1-Score']:.4f}")

print("\n⚡ Fastest Vectorization Method:")
fastest_vec = min(vectorization_times, key=vectorization_times.get)
print(f"   → {fastest_vec} with time: {vectorization_times[fastest_vec]:.2f}s")

print("\n💡 Key Observations:")
for name in results.keys():
    print(f"\n   {name}:")
    print(f"   - Accuracy:  {results[name]['Accuracy']:.4f}")
    print(f"   - Precision: {results[name]['Precision']:.4f}")
    print(f"   - Recall:    {results[name]['Recall']:.4f}")
    print(f"   - F1-Score:  {results[name]['F1-Score']:.4f}")
    print(f"   - Total Time: {results[name]['Total Time']:.2f}s")
    print(f"   - Features:  {vocabulary_sizes[name]:,}")

print("\n" + "=" * 60)
print("✅ EXPERIMENT COMPLETE! RESULTS READY FOR REPORT")
print("=" * 60)
