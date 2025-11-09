#!/bin/bash
# Setup script for Twitter Analysis Project

echo "Setting up Twitter Multimodal Analysis Project..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # For Linux/Mac
# For Windows: venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"

# Create sample data
echo "Generating sample data..."
python scripts/generate_sample_data.py --num-tweets 100 --output data/sample_tweets.json

echo "Setup completed successfully!"
echo "To run the analysis, use: python src/main.py --input data/sample_tweets.json --output results/"
