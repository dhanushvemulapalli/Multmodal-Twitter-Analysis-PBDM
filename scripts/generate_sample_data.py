"""
Sample Data Generator
Creates sample Twitter data in JSON format for testing
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_sample_tweets(num_tweets=100, output_path="data/sample_tweets.json"):
    """
    Generate sample Twitter data
    
    Args:
        num_tweets: Number of tweets to generate
        output_path: Path to save the JSON file
    """
    # Sample tweet templates
    positive_tweets = [
        "Amazing product! Really love it #great #amazing",
        "This is fantastic news! Verified information from reliable source",
        "So happy about this! Confirmed by official sources",
        "Excellent service, highly recommended!",
        "Great news! Fact-checked and confirmed"
    ]
    
    negative_tweets = [
        "Terrible experience, very disappointed #bad",
        "This is concerning, unverified claims circulating",
        "Fake news alert! Misinformation spreading",
        "Very poor quality, not recommended",
        "Unconfirmed rumor, be careful with this"
    ]
    
    neutral_tweets = [
        "Interesting article about technology",
        "Just sharing this information",
        "New update available now",
        "Checking out this new feature",
        "Regular update on the situation"
    ]
    
    users = [
        {"screen_name": "user1", "followers_count": 10000},
        {"screen_name": "user2", "followers_count": 5000},
        {"screen_name": "user3", "followers_count": 1000},
        {"screen_name": "user4", "followers_count": 500},
        {"screen_name": "user5", "followers_count": 100}
    ]
    
    tweets = []
    base_time = datetime.now()
    
    for i in range(num_tweets):
        # Random selection
        sentiment_type = random.choice(["positive", "negative", "neutral"])
        user = random.choice(users)
        
        if sentiment_type == "positive":
            text = random.choice(positive_tweets)
        elif sentiment_type == "negative":
            text = random.choice(negative_tweets)
        else:
            text = random.choice(neutral_tweets)
        
        # Add some variation
        text = text.replace("this", f"item_{i}")
        
        tweet = {
            "id": f"tweet_{i+1}",
            "text": text,
            "created_at": (base_time - timedelta(hours=random.randint(0, 24))).isoformat(),
            "user": user,
            "retweet_count": random.randint(0, 1000),
            "favorite_count": random.randint(0, 5000),
            "reply_count": random.randint(0, 100)
        }
        
        tweets.append(tweet)
    
    # Save to JSON file (JSON Lines format)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for tweet in tweets:
            f.write(json.dumps(tweet) + '\n')
    
    print(f"Generated {num_tweets} sample tweets and saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample Twitter data")
    parser.add_argument(
        "--num-tweets",
        type=int,
        default=100,
        help="Number of tweets to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample_tweets.json",
        help="Output file path"
    )
    
    args = parser.parse_args()
    generate_sample_tweets(args.num_tweets, args.output)
