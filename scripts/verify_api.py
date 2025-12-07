import os
import tweepy
from dotenv import load_dotenv
import sys

def verify_api():
    load_dotenv()
    
    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_SECRET")
    access_token = os.getenv("TWITTER_ACCESS_TOKEN")
    access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")

    print("Checking credentials presence...")
    print(f"API Key: {'Found' if api_key else 'Missing'}")
    print(f"Bearer Token: {'Found' if bearer_token else 'Missing'}")

    client = None
    
    # Try initializing Client
    try:
        if bearer_token:
            print("\nAttempting authentication with Bearer Token...")
            client = tweepy.Client(bearer_token=bearer_token)
        elif api_key and api_secret:
            print("\nAttempting authentication with API Key/Secret...")
            client = tweepy.Client(
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
        else:
            print("No credentials found!")
            return

        # Test 1: Get Me (User context) or simple user lookup (App context)
        print("\nTest 1: Verifying Authentication...")
        try:
            # For App-only (Bearer), we can't do get_me(), so let's try getting a user by username
            user = client.get_user(username="Twitter")
            if user.data:
                print(f"SUCCESS: Authenticated and fetched user: {user.data.name} (ID: {user.data.id})")
            else:
                print("WARNING: Authenticated but returned no data.")
        except tweepy.errors.TweepyException as e:
            print(f"FAILED: Authentication failed. Error: {e}")
            return

        # Test 2: Search Tweets
        print("\nTest 2: Verifying Search Access...")
        try:
            query = "python"
            response = client.search_recent_tweets(query=query, max_results=10)
            if response.data:
                print(f"SUCCESS: Fetched {len(response.data)} tweets.")
            else:
                print("SUCCESS: Search executed (but found 0 tweets).")
        except tweepy.errors.TweepyException as e:
            print(f"FAILED: Search failed. Error: {e}")
            if "403" in str(e):
                print("NOTE: 403 Forbidden usually means your Access Tier does not support this endpoint (e.g., Free Tier does not have Search access).")
            elif "429" in str(e):
                print("NOTE: 429 Too Many Requests means you have hit your rate limit.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    verify_api()
