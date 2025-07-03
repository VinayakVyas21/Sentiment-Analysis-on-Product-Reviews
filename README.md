# Sentiment Analysis on Product Reviews

This project performs sentiment analysis on product reviews using machine learning techniques. It includes exploratory data analysis, text preprocessing, feature engineering, model training, and evaluation.

## Project Overview

The project analyzes product reviews to determine whether they express positive or negative sentiment. It uses two different machine learning models (Logistic Regression and Random Forest) and compares their performance.

## Features

- **Data Exploration**: Analyzes class distribution and visualizes the most common words in positive and negative reviews using word clouds.
- **Text Preprocessing**: Converts text to lowercase, removes punctuation and stopwords, tokenizes, and applies stemming.
- **Feature Engineering**: Converts preprocessed text to TF-IDF features.
- **Model Training**: Trains Logistic Regression and Random Forest classifiers.
- **Model Evaluation**: Evaluates models using accuracy, precision, recall, F1 score, and confusion matrix.
- **Model Comparison**: Compares the performance of both models and identifies the best one.
- **Sample Predictions**: Predicts sentiment on sample reviews to demonstrate the model's capabilities.

## Dataset

The dataset (`reviews.csv`) contains product reviews with two columns:
- `review_text`: The text content of the review
- `sentiment`: The sentiment label (positive or negative)

## Requirements

To run this project, you need the following Python packages:

```
pandas
numpy
matplotlib
seaborn
wordcloud
nltk
scikit-learn
```

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn
```

## Usage

1. Make sure you have all the required packages installed.
2. Ensure the `reviews.csv` file is in the same directory as the script.
3. Run the script:

```bash
python sentiment_analysis.py
```

## Output

The script will generate several output files:

- `sentiment_distribution.png`: Bar chart showing the distribution of positive and negative reviews.
- `positive_wordcloud.png`: Word cloud of the most common words in positive reviews.
- `negative_wordcloud.png`: Word cloud of the most common words in negative reviews.
- `logistic_regression_confusion_matrix.png`: Confusion matrix for the Logistic Regression model.
- `random_forest_confusion_matrix.png`: Confusion matrix for the Random Forest model.
- `model_comparison.png`: Bar chart comparing the performance metrics of both models.

## Results

The script will output the performance metrics (accuracy, precision, recall, F1 score) for both models and identify the best model based on the F1 score. It will also predict the sentiment for five sample reviews to demonstrate the model's capabilities.

## Future Improvements

1. Collect more diverse training data
2. Experiment with more advanced NLP techniques like word embeddings (Word2Vec, GloVe)
3. Try more complex models such as deep learning approaches (LSTM, BERT)
4. Perform hyperparameter tuning to optimize model performance
5. Implement cross-validation for more robust evaluation
6. Add feature for aspect-based sentiment analysis to identify specific product aspects mentioned