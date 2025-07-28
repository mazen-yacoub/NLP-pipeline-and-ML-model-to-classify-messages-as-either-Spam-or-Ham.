# NLP-pipeline-and-ML-model-to-classify-messages-as-either-Spam-or-Ham.
This project presents a complete Natural Language Processing (NLP) pipeline for SMS Spam Detection using the popular SMS Spam Collection Dataset. It walks through essential stages of preprocessing, feature engineering, modeling, evaluation, and embeddings to classify messages as either Spam or Ham.

## Key Components

### 1. Data Loading & Exploration
- Dataset: `spam.csv` from the SMS Spam Collection (source: Kaggle)
- Cleaned unnecessary columns and formatted the labels
- Visualized label distribution and message length statistics

### 2. Text Preprocessing
Techniques used:
- Lowercasing
- Removing special characters and digits
- Tokenization
- Stopword removal
- Stemming using PorterStemmer
- Lemmatization using WordNetLemmatizer
- Created multiple versions of cleaned text for comparison

### 3. Feature Extraction
Methods used:
- Bag of Words (BoW)
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Word2Vec embeddings using Gensim
- Compared top words contributing to spam and ham classification

### 4. Document Similarity
- Applied cosine similarity using TF-IDF vectors
- Identified semantically similar messages

### 5. Text Classification
Models implemented:
- Multinomial Naive Bayes
- Logistic Regression

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score

Comparison was performed across BoW, TF-IDF, and Word2Vec feature sets.

Best performing setup: Logistic Regression with TF-IDF (highest F1-Score)

### 6. Embeddings with Word2Vec
- Trained a custom Word2Vec model on tokenized messages
- Demonstrated semantic similarity between selected words
- Created document vectors by averaging word embeddings

### 7. RNN & Sequence Modeling (Conceptual)
- Discussed the importance of word order in natural language processing
- Explained Recurrent Neural Networks (RNNs), their benefits and limitations
- Introduced improvements such as LSTM and GRU

## Conclusion

Best Performing Setup:
- Logistic Regression + TF-IDF → F1-Score ≈ 0.96

Insights:
- TF-IDF outperformed BoW in capturing term importance
- Logistic Regression delivered robust results across feature sets
- Word2Vec showed potential but benefits from larger datasets
- Spam messages commonly contained words like "free", "win", "call"
- Dataset was imbalanced: approximately 86.6% ham and 13.4% spam
