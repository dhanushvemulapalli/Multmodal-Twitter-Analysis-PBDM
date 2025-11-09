"""
Standalone Twitter Analysis without PySpark
Produces the same analysis results using regular Python/pandas
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import math
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import FACT_CHECK_KEYWORDS, FACTUALITY_THRESHOLDS, SENTIMENT_THRESHOLDS


def load_json_data(file_path: str) -> List[Dict]:
    """Load Twitter data from JSON file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # If no lines, try loading as standard JSON array
    if not data:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                pass
    
    return data


def preprocess_tweet(tweet: Dict) -> Dict:
    """Preprocess a single tweet"""
    import re
    
    processed = {
        'tweet_id': tweet.get('id', ''),
        'tweet_text': tweet.get('text', ''),
        'created_at': tweet.get('created_at', ''),
    }
    
    # Handle user data
    user = tweet.get('user', {})
    if isinstance(user, str):
        processed['username'] = user
        processed['followers_count'] = 0
    else:
        processed['username'] = user.get('screen_name', user.get('username', ''))
        processed['followers_count'] = user.get('followers_count', 0)
    
    # Engagement metrics
    processed['retweet_count'] = tweet.get('retweet_count', 0)
    processed['favorite_count'] = tweet.get('favorite_count', 0)
    processed['reply_count'] = tweet.get('reply_count', 0)
    
    # Clean text
    text = processed['tweet_text']
    text = re.sub(r'[^\w\s@#]|http\S+|www\.\S+', '', text.lower().strip())
    processed['tweet_text'] = ' '.join(text.split())
    
    # Text statistics
    processed['text_length'] = len(processed['tweet_text'])
    processed['word_count'] = len(processed['tweet_text'].split())
    processed['total_engagement'] = (
        processed['retweet_count'] + 
        processed['favorite_count'] + 
        processed['reply_count']
    )
    
    # Hashtag and mention counts
    processed['hashtag_count'] = processed['tweet_text'].count('#')
    processed['mention_count'] = processed['tweet_text'].count('@')
    
    return processed


def analyze_sentiment(text: str, vader: SentimentIntensityAnalyzer) -> Dict:
    """Analyze sentiment for a single text"""
    if not text or text.strip() == "":
        return {
            'sentiment_label': 'neutral',
            'sentiment_compound': 0.0,
            'sentiment_polarity': 0.0,
            'sentiment_confidence': 0.0
        }
    
    # Get VADER scores
    vader_scores = vader.polarity_scores(text)
    compound = vader_scores['compound']
    
    # Get TextBlob polarity
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
    except:
        polarity = 0.0
    
    # Combine scores (weighted average)
    combined_score = (compound * 0.6) + (polarity * 0.4)
    
    # Determine label
    if combined_score >= SENTIMENT_THRESHOLDS['positive']:
        label = 'positive'
        confidence = abs(combined_score)
    elif combined_score <= SENTIMENT_THRESHOLDS['negative']:
        label = 'negative'
        confidence = abs(combined_score)
    else:
        label = 'neutral'
        confidence = 1 - abs(combined_score)
    
    return {
        'sentiment_label': label,
        'sentiment_compound': float(compound),
        'sentiment_polarity': float(polarity),
        'sentiment_confidence': float(confidence)
    }


def calculate_keyword_score(text: str) -> float:
    """Calculate factuality score based on keywords"""
    if not text:
        return 0.5
    
    text_lower = text.lower()
    
    positive_keywords = [
        "verified", "fact-check", "factual", "reliable source",
        "confirmed", "official", "authentic", "credible"
    ]
    
    negative_keywords = [
        "fake news", "misinformation", "disinformation",
        "unverified", "unconfirmed", "rumor", "alleged",
        "unsubstantiated", "hoax"
    ]
    
    positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
    negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
    
    score = (positive_count - negative_count) / max(len(text.split()), 1)
    normalized_score = (score + 1) / 2
    
    return min(max(normalized_score, 0.0), 1.0)


def calculate_user_credibility(followers: int, engagement: int) -> float:
    """Calculate user credibility score"""
    if followers > 0:
        follower_score = min(math.log10(followers + 1) / 7, 1.0)
    else:
        follower_score = 0.1
    
    if followers > 0:
        engagement_ratio = min(engagement / max(followers, 1), 0.1)
        engagement_score = min(engagement_ratio * 10, 1.0)
    else:
        engagement_score = 0.1
    
    credibility = (follower_score * 0.6) + (engagement_score * 0.4)
    return min(max(credibility, 0.0), 1.0)


def calculate_engagement_quality(retweets: int, favorites: int, replies: int) -> float:
    """Calculate engagement quality score"""
    total_engagement = retweets + favorites + replies
    
    if total_engagement == 0:
        return 0.5
    
    if replies > 0:
        favorite_reply_ratio = favorites / replies
    else:
        favorite_reply_ratio = favorites
    
    normalized_ratio = min(favorite_reply_ratio / 10, 1.0)
    volume_factor = min(math.log10(total_engagement + 1) / 5, 1.0)
    
    return (normalized_ratio * 0.5) + (volume_factor * 0.5)


def detect_factuality(tweet: Dict) -> Dict:
    """Detect factuality for a single tweet"""
    text = tweet.get('tweet_text', '')
    followers = tweet.get('followers_count', 0)
    retweets = tweet.get('retweet_count', 0)
    favorites = tweet.get('favorite_count', 0)
    replies = tweet.get('reply_count', 0)
    
    keyword_score = calculate_keyword_score(text)
    engagement = retweets + favorites + replies
    credibility_score = calculate_user_credibility(followers, engagement)
    engagement_score = calculate_engagement_quality(retweets, favorites, replies)
    
    # Weighted combination
    factuality_score = (
        keyword_score * 0.4 +
        credibility_score * 0.3 +
        engagement_score * 0.3
    )
    
    # Determine reliability label
    if factuality_score >= FACTUALITY_THRESHOLDS['high_reliability']:
        reliability_label = 'high'
    elif factuality_score >= FACTUALITY_THRESHOLDS['medium_reliability']:
        reliability_label = 'medium'
    else:
        reliability_label = 'low'
    
    return {
        'factuality_score': float(factuality_score),
        'reliability_label': reliability_label,
        'keyword_score': float(keyword_score),
        'credibility_score': float(credibility_score),
        'engagement_score': float(engagement_score)
    }


def print_statistics(df: pd.DataFrame):
    """Print statistics about the analysis"""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Tweets: {len(df):,}")
    print(f"Unique Users: {df['username'].nunique():,}")
    print(f"Average Text Length: {df['text_length'].mean():.2f}")
    print(f"Average Word Count: {df['word_count'].mean():.2f}")
    print(f"Total Engagement: {df['total_engagement'].sum():,.0f}")
    print(f"Average Retweets: {df['retweet_count'].mean():.2f}")
    print(f"Average Favorites: {df['favorite_count'].mean():.2f}")
    
    print("\n" + "=" * 60)
    print("SENTIMENT DISTRIBUTION")
    print("=" * 60)
    sentiment_counts = df['sentiment_label'].value_counts()
    for label, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label.capitalize()}: {count:,} ({percentage:.2f}%)")
    
    print("\n" + "=" * 60)
    print("FACTUALITY/RELIABILITY DISTRIBUTION")
    print("=" * 60)
    reliability_counts = df['reliability_label'].value_counts()
    for label, count in reliability_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label.capitalize()}: {count:,} ({percentage:.2f}%)")
    
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"Average Sentiment Score: {df['sentiment_compound'].mean():.3f}")
    print(f"Average Factuality Score: {df['factuality_score'].mean():.3f}")
    print(f"Average Sentiment Confidence: {df['sentiment_confidence'].mean():.3f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Twitter Multimodal Analysis (without PySpark)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Twitter data file (JSON format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Path to output directory for results"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Twitter Multimodal Analysis (No Spark Version)")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Initialize NLTK
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
    except:
        pass
    
    # Load data
    print("\n[1/4] Loading data...")
    raw_data = load_json_data(str(input_path))
    print(f"✓ Loaded {len(raw_data)} tweets")
    
    # Preprocess
    print("\n[2/4] Preprocessing data...")
    processed_data = [preprocess_tweet(tweet) for tweet in raw_data]
    processed_data = [t for t in processed_data if t['tweet_text']]  # Remove empty tweets
    print(f"✓ Processed {len(processed_data)} tweets")
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Sentiment analysis
    print("\n[3/4] Performing sentiment analysis...")
    vader = SentimentIntensityAnalyzer()
    sentiment_results = [analyze_sentiment(row['tweet_text'], vader) for _, row in df.iterrows()]
    
    for i, result in enumerate(sentiment_results):
        for key, value in result.items():
            df.at[i, key] = value
    
    print("✓ Sentiment analysis completed")
    
    # Factuality detection
    print("\n[4/4] Performing factuality detection...")
    factuality_results = [detect_factuality(row.to_dict()) for _, row in df.iterrows()]
    
    for i, result in enumerate(factuality_results):
        for key, value in result.items():
            df.at[i, key] = value
    
    print("✓ Factuality detection completed")
    
    # Print statistics
    print_statistics(df)
    
    # Save results
    output_file = output_dir / "analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    # Display sample results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (First 10 tweets)")
    print("=" * 60)
    display_cols = ['tweet_id', 'tweet_text', 'sentiment_label', 'sentiment_compound', 
                    'reliability_label', 'factuality_score']
    print(df[display_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

