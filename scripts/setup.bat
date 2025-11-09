@echo off
REM Setup script for Twitter Analysis Project (Windows)

echo Setting up Twitter Multimodal Analysis Project...

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"

REM Create sample data
echo Generating sample data...
python scripts\generate_sample_data.py --num-tweets 100 --output data\sample_tweets.json

echo Setup completed successfully!
echo To run the analysis, use: python src\main.py --input data\sample_tweets.json --output results\
