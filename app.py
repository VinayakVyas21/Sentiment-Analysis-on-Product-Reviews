# Sentiment Analysis Web Interface
# This script creates a web interface for the sentiment analysis project using Streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis on Product Reviews",
    page_icon="😊",
    layout="wide"
)

# Function to preprocess text (same as in sentiment_analysis.py)
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
    
    # Remove punctuation but preserve exclamation marks and question marks
    modified_punctuation = string.punctuation.replace('!', '').replace('?', '')
    text = text.translate(str.maketrans('', '', modified_punctuation))
    
    # Replace multiple exclamation marks with a single one
    text = re.sub(r'!+', ' exclamation_mark_positive ', text)
    text = re.sub(r'\?+', ' question_mark ', text)
    
    # Add spaces around punctuation to treat them as separate tokens
    text = re.sub(r'([!?])', r' \1 ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords but keep sentiment-bearing words
    stop_words = set(stopwords.words('english'))
    sentiment_words = {'no', 'not', 'very', 'too', 'only', 'but', 'best', 'worst', 
                       'good', 'bad', 'fine', 'better', 'worse', 'expected'}
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

# Function to load models and vectorizer
@st.cache_resource
def load_models():
    # Check if models exist, if not, run the sentiment_analysis.py script
    if not os.path.exists('vectorizer.pkl') or not os.path.exists('best_model.pkl'):
        st.warning("Models not found. Please run sentiment_analysis.py first to train the models.")
        return None, None
    
    # Load vectorizer and model
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return vectorizer, model

# Function to predict sentiment
def predict_sentiment(review_text, vectorizer, model):
    # Preprocess the review
    processed_text = preprocess_text(review_text)
    
    # Transform to TF-IDF features
    features = vectorizer.transform([processed_text])
    
    # Get model prediction
    model_prediction = model.predict(features)[0]
    
    # Apply custom rules
    final_prediction = custom_rule_classifier(review_text, model_prediction)
    
    return final_prediction, processed_text

# Function to load and display dataset
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('reviews.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to create and display word clouds
def create_wordcloud(df, sentiment_type):
    text = ' '.join(df[df['sentiment'] == sentiment_type]['processed_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Most Common Words in {sentiment_type.capitalize()} Reviews')
    
    return fig

# Main function
def main():
    # Add a title and description
    st.title("Sentiment Analysis on Product Reviews")
    st.markdown("""
    This application analyzes product reviews to determine whether they express positive or negative sentiment.
    You can input your own review to get a sentiment prediction, or explore the dataset and visualizations.
    """)
    
    # Load models and vectorizer
    vectorizer, model = load_models()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Predict Sentiment", "Explore Dataset", "Visualizations"])
    
    # Tab 1: Predict Sentiment
    with tab1:
        st.header("Predict Sentiment")
        st.markdown("Enter a product review to predict its sentiment.")
        
        # Text input for user review
        user_review = st.text_area("Enter your review:", height=150)
        
        # Predict button
        if st.button("Predict Sentiment"):
            if not user_review:
                st.warning("Please enter a review.")
            elif vectorizer is None or model is None:
                st.error("Models not loaded. Please run sentiment_analysis.py first.")
            else:
                # Make prediction
                prediction, processed_text = predict_sentiment(user_review, vectorizer, model)
                
                # Display result with appropriate styling
                if prediction == "positive":
                    st.success(f"Sentiment: {prediction.upper()}")
                else:
                    st.error(f"Sentiment: {prediction.upper()}")
                
                # Show preprocessing details in an expander
                with st.expander("See preprocessing details"):
                    st.write("Original text:")
                    st.write(user_review)
                    st.write("Processed text:")
                    st.write(processed_text)
    
    # Tab 2: Explore Dataset
    with tab2:
        st.header("Explore Dataset")
        
        # Load dataset
        df = load_dataset()
        
        if df is not None:
            # Add processed text column if not already present
            if 'processed_text' not in df.columns:
                df['processed_text'] = df['review_text'].apply(preprocess_text)
            
            # Display dataset info
            st.subheader("Dataset Information")
            st.write(f"Number of reviews: {df.shape[0]}")
            st.write(f"Positive reviews: {sum(df['sentiment'] == 'positive')}")
            st.write(f"Negative reviews: {sum(df['sentiment'] == 'negative')}")
            
            # Display sample reviews
            st.subheader("Sample Reviews")
            st.dataframe(df.head(10))
            
            # Allow filtering by sentiment
            st.subheader("Filter Reviews")
            sentiment_filter = st.selectbox("Select sentiment:", ["All", "positive", "negative"])
            
            if sentiment_filter != "All":
                filtered_df = df[df['sentiment'] == sentiment_filter]
                st.dataframe(filtered_df)
    
    # Tab 3: Visualizations
    with tab3:
        st.header("Visualizations")
        
        # Load dataset
        df = load_dataset()
        
        if df is not None:
            # Add processed text column if not already present
            if 'processed_text' not in df.columns:
                df['processed_text'] = df['review_text'].apply(preprocess_text)
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(x='sentiment', data=df, ax=ax)
            ax.set_title('Sentiment Distribution')
            st.pyplot(fig)
            
            # Word clouds
            st.subheader("Word Clouds")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Positive Reviews Word Cloud**")
                positive_wordcloud = create_wordcloud(df, 'positive')
                st.pyplot(positive_wordcloud)
            
            with col2:
                st.markdown("**Negative Reviews Word Cloud**")
                negative_wordcloud = create_wordcloud(df, 'negative')
                st.pyplot(negative_wordcloud)
            
            # Load and display model comparison if available
            if os.path.exists('model_comparison.png'):
                st.subheader("Model Comparison")
                st.image('model_comparison.png')
            
            # Load and display confusion matrices if available
            st.subheader("Confusion Matrices")
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists('logistic_regression_confusion_matrix.png'):
                    st.markdown("**Logistic Regression Confusion Matrix**")
                    st.image('logistic_regression_confusion_matrix.png')
            
            with col2:
                if os.path.exists('random_forest_confusion_matrix.png'):
                    st.markdown("**Random Forest Confusion Matrix**")
                    st.image('random_forest_confusion_matrix.png')

# Run the app
if __name__ == "__main__":
    main()