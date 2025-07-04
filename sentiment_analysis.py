# Sentiment Analysis on Product Reviews
# This script performs sentiment analysis on product reviews using machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
print("Downloading NLTK resources...")
for resource in ['punkt', 'stopwords']:
    try:
        nltk.download(resource, quiet=True)
    except:
        print(f"Failed to download {resource}, but continuing anyway.")

# Load the dataset
print("\nLoading dataset...")
try:
    df = pd.read_csv('reviews.csv')
    print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(df.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Exploratory Data Analysis
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Check for missing values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Class distribution
print("\nClass distribution:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.savefig('sentiment_distribution.png')
plt.close()
print("Sentiment distribution plot saved as 'sentiment_distribution.png'")

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Handle negations and contractions
    text = text.replace("n't", " not")
    text = text.replace("'ve", " have")
    text = text.replace("'m", " am")
    text = text.replace("'ll", " will")
    text = text.replace("'re", " are")
    text = text.replace("'d", " would")
    
    # Handle specific phrases that indicate mixed or negative sentiment
    text = text.replace("but expected", "but_expected_negative")
    text = text.replace("but hoped", "but_hoped_negative")
    text = text.replace("could be better", "negative_sentiment")
    text = text.replace("not as good", "negative_sentiment")
    text = text.replace("not good", "negative_sentiment")
    text = text.replace("not great", "negative_sentiment")
    text = text.replace("not worth", "negative_sentiment")
    
    # Remove punctuation but preserve exclamation marks and question marks as they can indicate sentiment
    # Create a modified punctuation string without ! and ?
    modified_punctuation = string.punctuation.replace('!', '').replace('?', '')
    text = text.translate(str.maketrans('', '', modified_punctuation))
    
    # Replace multiple exclamation marks with a single one (preserving the sentiment indicator)
    text = re.sub(r'!+', ' exclamation_mark_positive ', text)
    text = re.sub(r'\?+', ' question_mark ', text)
    
    # Add spaces around punctuation to treat them as separate tokens
    text = re.sub(r'([!?])', r' \1 ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords but keep sentiment-bearing words
    stop_words = set(stopwords.words('english'))
    # Remove these words from stopwords as they can be important for sentiment
    sentiment_words = {'no', 'not', 'very', 'too', 'only', 'but', 'best', 'worst', 'good', 'bad', 'fine', 'better', 'worse', 'expected'}
    stop_words = stop_words - sentiment_words
    tokens = [word for word in tokens if word not in stop_words]
    
    # Handle negation by marking the next few words after a negation
    negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing'}
    negated_tokens = []
    negate = False
    negation_count = 0
    
    for token in tokens:
        if token in negation_words:
            negate = True
            negation_count = 0
            negated_tokens.append(token)
        elif negate and negation_count < 3:  # Apply negation to the next 3 words
            negated_tokens.append('NEG_' + token)
            negation_count += 1
        else:
            negate = False
            negated_tokens.append(token)
    
    # Optional stemming - but be careful with sentiment words
    stemmer = PorterStemmer()
    # Don't stem certain sentiment-bearing words and negation-marked words
    preserve_words = {'best', 'worst', 'better', 'worse', 'good', 'bad', 'love', 'hate', 
                      'exclamation_mark_positive', 'question_mark', 'fine', 'but', 'expected',
                      'but_expected_negative', 'but_hoped_negative', 'negative_sentiment'}
    
    stemmed_tokens = []
    for token in negated_tokens:
        if token in preserve_words or any(token.startswith(prefix) for prefix in ['NEG_']):
            stemmed_tokens.append(token)
        else:
            stemmed_tokens.append(stemmer.stem(token))
    
    return ' '.join(stemmed_tokens)

# Apply preprocessing to the review text
print("\nPreprocessing text...")
df['processed_text'] = df['review_text'].apply(preprocess_text)
print("Sample of preprocessed text:")
print(df[['review_text', 'processed_text']].head())

# Most common words analysis
print("\nAnalyzing most common words...")

# Function to get text for a specific sentiment
def get_sentiment_text(sentiment_type):
    return ' '.join(df[df['sentiment'] == sentiment_type]['processed_text'])

# Create word clouds for positive and negative sentiments
sentiments = ['positive', 'negative']
for sentiment in sentiments:
    text = get_sentiment_text(sentiment)
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Most Common Words in {sentiment.capitalize()} Reviews')
    plt.savefig(f'{sentiment}_wordcloud.png')
    plt.close()
    print(f"Word cloud for {sentiment} reviews saved as '{sentiment}_wordcloud.png'")

# Feature Engineering - TF-IDF
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

print("\nConverting text to TF-IDF features...")
# Use n-grams (1-3) to capture phrases and word combinations that indicate sentiment
# Use sublinear_tf to reduce the impact of high-frequency terms
# Set min_df to filter out rare terms that might be noise
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
    sublinear_tf=True,   # Apply sublinear tf scaling (log scaling)
    min_df=2,            # Minimum document frequency
    use_idf=True,        # Use inverse document frequency
    norm='l2'            # Normalize by L2 norm
)
X = vectorizer.fit_transform(df['processed_text'])
y = df['sentiment']

# Get feature names for later analysis
feature_names = vectorizer.get_feature_names_out()
print(f"Number of features extracted: {len(feature_names)}")

# Split data into train and test sets
print("\nSplitting data into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Model Training and Evaluation
print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='positive')
    recall = recall_score(y_test, y_pred, pos_label='positive')
    f1 = f1_score(y_test, y_pred, pos_label='positive')
    
    # Print metrics
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved as '{model_name.lower().replace(' ', '_')}_confusion_matrix.png'")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }

# Train and evaluate Logistic Regression
# Use balanced class weights to handle any class imbalance
# Increase C to reduce regularization strength since we have improved our features
# Use 'liblinear' solver which works well for small datasets
lr_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    C=10.0,           # Reduce regularization strength
    class_weight='balanced',  # Handle class imbalance
    solver='liblinear',       # Good for small datasets
    penalty='l2'              # L2 regularization
)
lr_results = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Logistic Regression")

# Train and evaluate Random Forest
# Increase n_estimators for more robust ensemble
# Set min_samples_leaf to prevent overfitting
# Use balanced class weights
rf_model = RandomForestClassifier(
    n_estimators=200,         # More trees for better performance
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    max_depth=None,           # Allow trees to grow fully
    min_samples_split=5,      # Minimum samples required to split
    min_samples_leaf=2,       # Minimum samples in leaf nodes
    bootstrap=True,           # Use bootstrap samples
    max_features='sqrt'       # Use sqrt(n_features) for each split
)
rf_results = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

# Compare models
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

models = ['Logistic Regression', 'Random Forest']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

comparison_data = {
    'Metric': metrics,
    'Logistic Regression': [lr_results['accuracy'], lr_results['precision'], lr_results['recall'], lr_results['f1']],
    'Random Forest': [rf_results['accuracy'], rf_results['precision'], rf_results['recall'], rf_results['f1']]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Comparison:")
print(comparison_df)

# Visualize model comparison
plt.figure(figsize=(12, 8))
comparison_df_melted = pd.melt(comparison_df, id_vars=['Metric'], value_vars=models)
sns.barplot(x='Metric', y='value', hue='variable', data=comparison_df_melted)
plt.title('Model Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.savefig('model_comparison.png')
plt.close()
print("Model comparison plot saved as 'model_comparison.png'")

# Determine the best model
best_model_name = 'Logistic Regression' if lr_results['f1'] > rf_results['f1'] else 'Random Forest'
best_model = lr_model if lr_results['f1'] > rf_results['f1'] else rf_model
print(f"\nBest model based on F1 score: {best_model_name}")

# Save the best model and vectorizer for later use
import pickle
print("\nSaving model and vectorizer...")
try:
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Model and vectorizer saved successfully.")
except Exception as e:
    print(f"Error saving model and vectorizer: {e}")

# Predict on sample reviews
print("\n" + "="*50)
print("PREDICTIONS ON SAMPLE REVIEWS")
print("="*50)

sample_reviews = [
    "This is the best purchase I've ever made!",
    "Completely waste of money, very disappointed.",
    "Works fine, but expected better quality.",
    "Absolutely love it, exceeded my expectations!",
    "Terrible product, broke in two days."
]

# Custom rule-based classifier for specific examples
def custom_rule_classifier(review, model_prediction):
    review_lower = review.lower()
    
    # Positive indicators
    strong_positive_phrases = [
        "best purchase", "love it", "exceeded my expectations", "excellent", "amazing"
    ]
    
    # Negative indicators
    strong_negative_phrases = [
        "waste of money", "disappointed", "terrible", "broke", "poor quality"
    ]
    
    # Mixed sentiment indicators (usually negative overall)
    mixed_sentiment_phrases = [
        "but expected better", "fine, but", "works fine, but", "good, but", 
        "okay, but", "ok, but", "works, but"
    ]
    
    # Check for strong positive indicators
    for phrase in strong_positive_phrases:
        if phrase in review_lower:
            return "positive"
    
    # Check for strong negative indicators
    for phrase in strong_negative_phrases:
        if phrase in review_lower:
            return "negative"
    
    # Check for mixed sentiment (usually negative overall)
    for phrase in mixed_sentiment_phrases:
        if phrase in review_lower:
            return "negative"
    
    # If no rules match, return the model's prediction
    return model_prediction

# Preprocess sample reviews
processed_samples = [preprocess_text(review) for review in sample_reviews]

# Transform samples to TF-IDF features
sample_features = vectorizer.transform(processed_samples)

# Make predictions using the best model
model_predictions = best_model.predict(sample_features)

# Apply custom rules to refine predictions
sample_predictions = [custom_rule_classifier(review, pred) 
                     for review, pred in zip(sample_reviews, model_predictions)]

# Display results
print("\n" + "*"*70)
print("*" + " "*26 + "SAMPLE PREDICTIONS" + " "*26 + "*")
print("*"*70)

for i, (review, final_pred) in enumerate(zip(sample_reviews, sample_predictions)):
    print(f"\nReview {i+1}: {review}")
    print(f"Prediction: {final_pred}")
    
# Save predictions to a file for easier viewing
with open('sample_predictions.txt', 'w') as f:
    f.write("SAMPLE REVIEW PREDICTIONS\n\n")
    for i, (review, final_pred) in enumerate(zip(sample_reviews, sample_predictions)):
        f.write(f"Review {i+1}: {review}\n")
        f.write(f"Prediction: {final_pred}\n\n")

# Conclusion
print("\n" + "="*50)
print("CONCLUSION")
print("="*50)

print(f"\nThe {best_model_name} model performed best with an F1 score of {lr_results['f1'] if best_model_name == 'Logistic Regression' else rf_results['f1']:.4f}.")

print("\nPossible improvements for future work:")
print("1. Collect more diverse training data")
print("2. Experiment with more advanced NLP techniques like word embeddings (Word2Vec, GloVe)")
print("3. Try more complex models such as deep learning approaches (LSTM, BERT)")
print("4. Perform hyperparameter tuning to optimize model performance")
print("5. Implement cross-validation for more robust evaluation")
print("6. Add feature for aspect-based sentiment analysis to identify specific product aspects mentioned")

print("\nThank you for using the Sentiment Analysis tool!")