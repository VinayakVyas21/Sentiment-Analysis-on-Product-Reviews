# Sentiment Analysis Web Interface

This is a web interface for the Sentiment Analysis on Product Reviews project. It allows users to interact with the sentiment analysis model through a user-friendly interface.

## Features

- **Predict Sentiment**: Enter your own product review and get a sentiment prediction in real-time
- **Explore Dataset**: View and filter the dataset of product reviews
- **Visualizations**: Explore visualizations including sentiment distribution, word clouds, and model performance metrics

## Prerequisites

Before running the web interface, make sure you have:

1. Run the `sentiment_analysis.py` script to train the model and generate necessary files
2. Installed all required dependencies

## Installation

1. Install the required packages:

```bash
pip install -r requirements_app.txt
```

## Running the Web Interface

To start the web interface, run:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Usage

### Predict Sentiment

1. Navigate to the "Predict Sentiment" tab
2. Enter a product review in the text area
3. Click the "Predict Sentiment" button
4. View the prediction result and preprocessing details

### Explore Dataset

1. Navigate to the "Explore Dataset" tab
2. View dataset information and sample reviews
3. Filter reviews by sentiment using the dropdown menu

### Visualizations

1. Navigate to the "Visualizations" tab
2. Explore various visualizations including:
   - Sentiment distribution
   - Word clouds for positive and negative reviews
   - Model comparison
   - Confusion matrices

## Troubleshooting

- If you encounter an error about missing models, make sure you've run `sentiment_analysis.py` first
- If the application fails to start, check that all dependencies are installed correctly
- For any other issues, refer to the Streamlit documentation or open an issue in the project repository