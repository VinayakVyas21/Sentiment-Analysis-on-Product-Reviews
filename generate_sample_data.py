# Generate Sample Dataset for Sentiment Analysis
# This script creates a larger sample dataset of product reviews with sentiment labels

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define sample positive and negative phrases to generate reviews
positive_phrases = [
    "love this product", "excellent quality", "works perfectly", "highly recommend",
    "great value", "exceeded expectations", "very satisfied", "best purchase",
    "amazing product", "fantastic", "worth every penny", "impressive",
    "very pleased", "outstanding", "perfect fit", "great features",
    "easy to use", "high quality", "excellent service", "very happy"
]

negative_phrases = [
    "waste of money", "poor quality", "doesn't work", "disappointed",
    "terrible product", "broke quickly", "not worth the price", "avoid this",
    "regret buying", "cheaply made", "stopped working", "frustrating",
    "poor design", "defective", "not as described", "uncomfortable",
    "difficult to use", "low quality", "poor customer service", "unhappy"
]

# Function to generate a random review
def generate_review(sentiment):
    phrases = positive_phrases if sentiment == 'positive' else negative_phrases
    
    # Select 1-3 phrases randomly
    num_phrases = random.randint(1, 3)
    selected_phrases = random.sample(phrases, num_phrases)
    
    # Add some filler words and combine phrases
    fillers = ["I", "The product", "It", "This item", "This product"]
    verbs = ["is", "was", "has", "provides", "gives"]
    adverbs = ["really", "very", "extremely", "surprisingly", ""]
    
    review = []
    for phrase in selected_phrases:
        if random.random() < 0.7:  # 70% chance to add a structured sentence
            filler = random.choice(fillers)
            verb = random.choice(verbs)
            adverb = random.choice(adverbs)
            
            if adverb:
                review.append(f"{filler} {verb} {adverb} {phrase}.")
            else:
                review.append(f"{filler} {verb} {phrase}.")
        else:
            review.append(f"{phrase.capitalize()}!")
    
    return " ".join(review)

# Generate dataset
def generate_dataset(num_samples=500, positive_ratio=0.5):
    num_positive = int(num_samples * positive_ratio)
    num_negative = num_samples - num_positive
    
    reviews = []
    sentiments = []
    
    # Generate positive reviews
    for _ in range(num_positive):
        reviews.append(generate_review('positive'))
        sentiments.append('positive')
    
    # Generate negative reviews
    for _ in range(num_negative):
        reviews.append(generate_review('negative'))
        sentiments.append('negative')
    
    # Create DataFrame
    df = pd.DataFrame({
        'review_text': reviews,
        'sentiment': sentiments
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

# Generate and save dataset
def main():
    print("Generating sample dataset...")
    df = generate_dataset(num_samples=500, positive_ratio=0.5)
    
    # Save to CSV
    output_file = 'generated_reviews.csv'
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file} with {len(df)} samples")
    print(f"Positive reviews: {sum(df['sentiment'] == 'positive')}")
    print(f"Negative reviews: {sum(df['sentiment'] == 'negative')}")
    
    # Display sample
    print("\nSample reviews:")
    print(df.head())

if __name__ == "__main__":
    main()