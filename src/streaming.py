"""
Twitter Streaming Module for Real-Time Tweet Processing
Uses Twitter API v2 filtered stream endpoint with PySpark Structured Streaming
"""
import os
import json
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    from twarc import Twarc2
    TWARC_AVAILABLE = True
except ImportError:
    TWARC_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TwitterStreamProducer:
    """
    Produces tweets from Twitter API v2 filtered stream
    Writes tweets to a directory for Spark Structured Streaming to consume
    """
    
    def __init__(
        self,
        keyword: str,
        output_dir: str = "stream_data",
        batch_interval: int = 3,
        max_retries: int = 5,
        retry_delay: int = 5
    ):
        """
        Initialize Twitter stream producer
        
        Args:
            keyword: Keyword to filter tweets
            output_dir: Directory to write streaming JSON files
            batch_interval: Interval in seconds between file writes
            max_retries: Maximum retry attempts for API calls
            retry_delay: Delay between retries in seconds
        """
        self.keyword = keyword
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_interval = batch_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get API credentials
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        
        # Streaming state
        self.running = False
        self.tweet_queue = queue.Queue()
        self.tweet_count = 0
        self.error_count = 0
        
        # Initialize API client
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Twitter API client (prefer Tweepy v2, fallback to Twarc2)"""
        if not self.bearer_token and not (self.api_key and self.api_secret):
            print("Warning: Twitter API credentials not found. Will use local file fallback.")
            return None
        
        # Prefer Tweepy v2 (simpler for filtered stream)
        if TWEEPY_AVAILABLE and self.bearer_token:
            try:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token,
                    wait_on_rate_limit=True
                )
                print("✓ Initialized Tweepy v2 client")
                return self.client
            except Exception as e:
                print(f"Warning: Failed to initialize Tweepy: {e}")
        
        # Fallback to Twarc2
        if TWARC_AVAILABLE:
            try:
                if self.bearer_token:
                    self.client = Twarc2(bearer_token=self.bearer_token)
                elif self.api_key and self.api_secret:
                    self.client = Twarc2(
                        consumer_key=self.api_key,
                        consumer_secret=self.api_secret,
                        access_token=self.access_token,
                        access_token_secret=self.access_token_secret
                    )
                if self.client:
                    print("✓ Initialized Twarc2 client")
                    return self.client
            except Exception as e:
                print(f"Warning: Failed to initialize Twarc2: {e}")
        
        return None
    
    def _format_tweet_for_spark(self, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format tweet data to match expected Spark schema
        
        Args:
            tweet_data: Raw tweet data from API
            
        Returns:
            Formatted tweet dictionary
        """
        # Handle both Tweepy and Twarc2 formats
        if isinstance(tweet_data, dict):
            # Tweepy v2 format
            if "data" in tweet_data:
                tweet = tweet_data["data"]
                user = tweet_data.get("includes", {}).get("users", [{}])[0] if "includes" in tweet_data else {}
            else:
                # Direct tweet dict or Twarc2 format
                tweet = tweet_data
                user = tweet.get("author", {}) or tweet.get("user", {})
        else:
            # Tweepy Response object
            tweet = tweet_data.data if hasattr(tweet_data, "data") else {}
            user = {}
        
        # Extract fields
        tweet_id = str(tweet.get("id", ""))
        text = tweet.get("text", "")
        created_at = tweet.get("created_at", datetime.utcnow().isoformat())
        
        # User data
        username = user.get("username", "") or user.get("screen_name", "")
        followers_count = user.get("public_metrics", {}).get("followers_count", 0) if isinstance(user.get("public_metrics"), dict) else user.get("followers_count", 0)
        
        # Engagement metrics
        public_metrics = tweet.get("public_metrics", {})
        if isinstance(public_metrics, dict):
            retweet_count = public_metrics.get("retweet_count", 0)
            favorite_count = public_metrics.get("like_count", 0) or public_metrics.get("favorite_count", 0)
            reply_count = public_metrics.get("reply_count", 0)
        else:
            retweet_count = tweet.get("retweet_count", 0)
            favorite_count = tweet.get("favorite_count", 0) or tweet.get("like_count", 0)
            reply_count = tweet.get("reply_count", 0)
        
        # Format for Spark ingestion
        formatted = {
            "id": tweet_id,
            "text": text,
            "created_at": created_at,
            "user": {
                "screen_name": username,
                "followers_count": followers_count
            },
            "retweet_count": retweet_count,
            "favorite_count": favorite_count,
            "reply_count": reply_count
        }
        
        return formatted
    
    def _stream_tweets_tweepy(self):
        """Stream tweets using Tweepy v2 filtered stream"""
        if not self.client or not self.bearer_token:
            return
        
        try:
            # Create StreamingClient (separate from Client)
            stream_client = tweepy.StreamingClient(
                bearer_token=self.bearer_token,
                wait_on_rate_limit=True
            )
            
            # Delete existing rules
            try:
                rules_response = stream_client.get_rules()
                if rules_response.data:
                    rule_ids = [rule.id for rule in rules_response.data]
                    stream_client.delete_rules(ids=rule_ids)
            except Exception as e:
                print(f"Note: Could not delete existing rules: {e}")
            
            # Add new rule
            try:
                rule_value = f"{self.keyword} lang:en"
                stream_client.add_rules(tweepy.StreamRule(value=rule_value))
                print(f"✓ Added stream rule: {rule_value}")
            except Exception as e:
                print(f"Warning: Could not add stream rule: {e}")
                # Try without lang filter
                try:
                    stream_client.add_rules(tweepy.StreamRule(value=self.keyword))
                    print(f"✓ Added stream rule: {self.keyword}")
                except:
                    print("Error: Could not add stream rules. Falling back to local file mode.")
                    return
            
            class TweetStreamListener(tweepy.StreamingClient):
                def __init__(self, producer):
                    super().__init__(bearer_token=producer.bearer_token, wait_on_rate_limit=True)
                    self.producer = producer
                
                def on_tweet(self, tweet):
                    try:
                        # Format tweet data
                        tweet_dict = {
                            "data": {
                                "id": str(tweet.id),
                                "text": tweet.text,
                                "created_at": tweet.created_at.isoformat() if hasattr(tweet, 'created_at') and tweet.created_at else datetime.utcnow().isoformat(),
                                "public_metrics": {
                                    "retweet_count": tweet.public_metrics.get("retweet_count", 0) if hasattr(tweet, 'public_metrics') and tweet.public_metrics else 0,
                                    "like_count": tweet.public_metrics.get("like_count", 0) if hasattr(tweet, 'public_metrics') and tweet.public_metrics else 0,
                                    "reply_count": tweet.public_metrics.get("reply_count", 0) if hasattr(tweet, 'public_metrics') and tweet.public_metrics else 0
                                }
                            },
                            "includes": {
                                "users": [{
                                    "username": getattr(tweet, 'author_id', '') if hasattr(tweet, 'author_id') else "",
                                    "public_metrics": {}
                                }]
                            }
                        }
                        formatted = self.producer._format_tweet_for_spark(tweet_dict)
                        self.producer.tweet_queue.put(formatted)
                        self.producer.tweet_count += 1
                    except Exception as e:
                        self.producer.error_count += 1
                        if self.producer.error_count % 10 == 0:
                            print(f"Error processing tweet: {e}")
                
                def on_connection_error(self):
                    print("Connection error in stream")
                    self.producer.error_count += 1
                    return False
                
                def on_request_error(self, status_code):
                    print(f"Request error: {status_code}")
                    self.producer.error_count += 1
                    return False
            
            listener = TweetStreamListener(self)
            # Start streaming with tweet fields
            listener.filter(
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"],
                user_fields=["username", "public_metrics"]
            )
            
        except Exception as e:
            print(f"Error in Tweepy stream: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _stream_tweets_twarc(self):
        """Stream tweets using Twarc2"""
        if not self.client:
            return
        
        try:
            # Use Twarc2 filtered stream
            for response in self.client.search_recent(
                query=f"{self.keyword} lang:en",
                max_results=100
            ):
                if "data" in response:
                    for tweet in response["data"]:
                        try:
                            formatted = self._format_tweet_for_spark(response)
                            self.tweet_queue.put(formatted)
                            self.tweet_count += 1
                        except Exception as e:
                            self.error_count += 1
                            if self.error_count % 10 == 0:
                                print(f"Error processing tweet: {e}")
                
                # Rate limiting
                time.sleep(1)
                
        except Exception as e:
            print(f"Error in Twarc stream: {e}")
            raise
    
    def _write_batch_to_file(self):
        """Write queued tweets to JSON file for Spark consumption"""
        batch = []
        batch_size = 0
        
        # Collect tweets from queue
        while not self.tweet_queue.empty() or batch_size < 10:
            try:
                tweet = self.tweet_queue.get(timeout=0.1)
                batch.append(tweet)
                batch_size += 1
            except queue.Empty:
                if batch:
                    break
                time.sleep(0.1)
        
        if batch:
            # Write to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = self.output_dir / f"tweets_{timestamp}.json"
            
            # Write as JSON Lines (one JSON object per line)
            with open(file_path, "w", encoding="utf-8") as f:
                for tweet in batch:
                    f.write(json.dumps(tweet) + "\n")
            
            return len(batch)
        return 0
    
    def _stream_worker(self):
        """Worker thread for streaming tweets"""
        retry_count = 0
        
        while self.running:
            try:
                if self.client and self.bearer_token:
                    # Try Tweepy v2 streaming
                    if TWEEPY_AVAILABLE:
                        self._stream_tweets_tweepy()
                    # Fallback to Twarc2
                    elif TWARC_AVAILABLE:
                        self._stream_tweets_twarc()
                    else:
                        print("No streaming library available. Using local file fallback.")
                        time.sleep(self.batch_interval)
                else:
                    # No API client - fallback to local file mode
                    print("No API client available. Using local file fallback.")
                    print("Place JSON files in stream_data/ directory for local streaming.")
                    time.sleep(self.batch_interval)
                    
            except Exception as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    print(f"Max retries exceeded. Error: {e}")
                    print("Falling back to local file streaming mode.")
                    print("Place JSON files in stream_data/ directory for local streaming.")
                    self.client = None
                    break
                
                print(f"Stream error (retry {retry_count}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay)
    
    def _file_writer_worker(self):
        """Worker thread for writing batches to files"""
        while self.running:
            try:
                written = self._write_batch_to_file()
                if written > 0:
                    print(f"✓ Wrote batch of {written} tweets (Total: {self.tweet_count})")
                time.sleep(self.batch_interval)
            except Exception as e:
                print(f"Error writing batch: {e}")
                time.sleep(self.batch_interval)
    
    def start(self):
        """Start streaming tweets"""
        if self.running:
            return
        
        self.running = True
        
        # Start streaming thread
        if self.client:
            stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            stream_thread.start()
        
        # Start file writer thread
        writer_thread = threading.Thread(target=self._file_writer_worker, daemon=True)
        writer_thread.start()
        
        print(f"Started Twitter stream for keyword: '{self.keyword}'")
        print(f"Output directory: {self.output_dir}")
        print(f"Batch interval: {self.batch_interval}s")
    
    def stop(self):
        """Stop streaming tweets"""
        self.running = False
        print(f"\nStopped streaming. Total tweets collected: {self.tweet_count}")


class LocalFileStreamProducer:
    """
    Fallback producer that reads from local streaming JSON files
    Used when Twitter API is unavailable
    """
    
    def __init__(self, input_dir: str = "stream_data", batch_interval: int = 3):
        """
        Initialize local file stream producer
        
        Args:
            input_dir: Directory containing streaming JSON files
            batch_interval: Interval between file checks
        """
        self.input_dir = Path(input_dir)
        self.batch_interval = batch_interval
        self.processed_files = set()
    
    def get_latest_files(self, limit: int = 10):
        """
        Get latest unprocessed JSON files
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of file paths
        """
        if not self.input_dir.exists():
            return []
        
        json_files = sorted(
            self.input_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        unprocessed = [f for f in json_files if str(f) not in self.processed_files]
        return unprocessed[:limit]
    
    def mark_processed(self, file_path: Path):
        """Mark a file as processed"""
        self.processed_files.add(str(file_path))

