"""
Factuality Detection Module
Assesses the factuality and reliability of tweets using ML-based and heuristic methods
"""
import math
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, lower, when


# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import FACT_CHECK_KEYWORDS, FACTUALITY_THRESHOLDS


# Try to import transformers for ML-based detection
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Will use heuristic-based factuality detection only.")


class FactualityDetector:
    """Detects factuality and reliability of Twitter content using ML and heuristics"""
    
    def __init__(self, spark: SparkSession, use_ml: bool = True, use_ml_for_streaming: bool = False):
        """
        Initialize factuality detector
        
        Args:
            spark: SparkSession instance
            use_ml: Whether to use ML-based detection (requires transformers)
            use_ml_for_streaming: Whether to use ML for streaming (can cause UDF serialization issues)
        """
        self.spark = spark
        self.fact_check_keywords = FACT_CHECK_KEYWORDS
        
        # ML mode disabled for streaming by default to avoid SparkContext serialization errors
        self.use_ml = use_ml and TRANSFORMERS_AVAILABLE and use_ml_for_streaming
        
        # Lazy load ML model
        self._ml_model = None
        self._ml_tokenizer = None
        
        if use_ml and not use_ml_for_streaming and TRANSFORMERS_AVAILABLE:
            print("ℹ ML-based factuality detection available but disabled for streaming mode")
            print("  (Use heuristic-based detection to avoid UDF serialization issues)")
        elif self.use_ml:
            print("✓ ML-based factuality detection enabled")
        else:
            print("✓ Heuristic-based factuality detection enabled")
    
    def _load_ml_model(self):
        """Lazy load the ML model for fake news detection"""
        if self._ml_model is None and self.use_ml:
            try:
                print("Loading ML factuality detection model (DistilBERT)...")
                model_name = "dhruvpal/fake-news-bert"
                self._ml_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._ml_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._ml_model.eval()  # Set to evaluation mode
                print("✓ ML model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load ML model: {e}")
                print("Falling back to heuristic-based detection")
                self.use_ml = False
    
    def _predict_ml_factuality(self, text: str) -> tuple:
        """
        Predict factuality using ML model
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (factuality_score, label) where label is 'real' or 'fake'
        """
        if not self.use_ml or not text:
            return (0.5, "unknown")
        
        self._load_ml_model()
        
        if self._ml_model is None:
            return (0.5, "unknown")
        
        try:
            # Tokenize and predict
            inputs = self._ml_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self._ml_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                # Model outputs: LABEL_0 = Fake, LABEL_1 = Real
                fake_prob = probabilities[0][0].item()
                real_prob = probabilities[0][1].item()
                
                # Factuality score: higher for real news
                factuality_score = real_prob
                label = "real" if real_prob > fake_prob else "fake"
                
                return (factuality_score, label)
        except Exception as e:
            print(f"Warning: ML prediction failed: {e}")
            return (0.5, "unknown")
    
    def _calculate_keyword_score(self, text: str) -> float:
        """
        Calculate score based on fact-checking keywords
        
        Args:
            text: Input text
            
        Returns:
            Keyword score (0-1)
        """
        if not text:
            return 0.5  # Neutral score for empty text
        
        text_lower = text.lower()
        
        # Positive indicators (increase reliability)
        positive_keywords = [
            "verified", "fact-check", "factual", "reliable source",
            "confirmed", "official", "authentic", "credible"
        ]
        
        # Negative indicators (decrease reliability)
        negative_keywords = [
            "fake news", "misinformation", "disinformation",
            "unverified", "unconfirmed", "rumor", "alleged",
            "unsubstantiated", "hoax"
        ]
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        # Calculate score: -1 to 1, normalized to 0-1
        score = (positive_count - negative_count) / max(len(text.split()), 1)
        normalized_score = (score + 1) / 2  # Normalize to 0-1
        
        return min(max(normalized_score, 0.0), 1.0)
    
    def _calculate_user_credibility_score(self, followers: int, engagement: int) -> float:
        """
        Calculate user credibility based on followers and engagement
        
        Args:
            followers: Number of followers
            engagement: Total engagement (retweets + favorites)
            
        Returns:
            Credibility score (0-1)
        """
        # Normalize followers (log scale for large differences)
        if followers > 0:
            follower_score = min(math.log10(followers + 1) / 7, 1.0)  # Cap at 10M followers
        else:
            follower_score = 0.1
        
        # Engagement ratio
        if followers > 0:
            engagement_ratio = min(engagement / max(followers, 1), 0.1)  # Normalize
            engagement_score = min(engagement_ratio * 10, 1.0)
        else:
            engagement_score = 0.1
        
        # Combined credibility (weighted)
        credibility = (follower_score * 0.6) + (engagement_score * 0.4)
        
        return min(max(credibility, 0.0), 1.0)
    
    def _calculate_engagement_quality(self, retweets: int, favorites: int, replies: int) -> float:
        """
        Assess engagement quality
        
        Args:
            retweets: Number of retweets
            favorites: Number of favorites
            replies: Number of replies
            
        Returns:
            Quality score (0-1)
        """
        total_engagement = retweets + favorites + replies
        
        if total_engagement == 0:
            return 0.5  # Neutral for no engagement
        
        # High favorites-to-replies ratio might indicate quality content
        if replies > 0:
            favorite_reply_ratio = favorites / replies
        else:
            favorite_reply_ratio = favorites
        
        # Normalize
        normalized_ratio = min(favorite_reply_ratio / 10, 1.0)
        
        # Engagement volume factor
        volume_factor = min(math.log10(total_engagement + 1) / 5, 1.0)
        
        return (normalized_ratio * 0.5) + (volume_factor * 0.5)
    
    def _calculate_factuality_score(
        self, text: str, followers: int, retweets: int,
        favorites: int, replies: int
    ) -> tuple:
        """
        Calculate overall factuality score using ML + heuristics
        
        Args:
            text: Tweet text
            followers: Follower count
            retweets: Retweet count
            favorites: Favorite count
            replies: Reply count
            
        Returns:
            Tuple of (factuality_score, reliability_label, keyword_score, credibility_score, engagement_score, ml_score)
        """
        # ML-based score
        ml_score = 0.5
        ml_label = "unknown"
        if self.use_ml and text:
            ml_score, ml_label = self._predict_ml_factuality(text)
        
        # Heuristic scores
        keyword_score = self._calculate_keyword_score(text)
        engagement = retweets + favorites + replies
        credibility_score = self._calculate_user_credibility_score(followers, engagement)
        engagement_score = self._calculate_engagement_quality(retweets, favorites, replies)
        
        # Weighted combination: prioritize ML if available
        if self.use_ml and ml_label != "unknown":
            factuality_score = (
                ml_score * 0.6 +           # ML model gets highest weight
                keyword_score * 0.2 +
                credibility_score * 0.1 +
                engagement_score * 0.1
            )
        else:
            # Fallback to heuristic-only
            factuality_score = (
                keyword_score * 0.4 +
                credibility_score * 0.3 +
                engagement_score * 0.3
            )
        
        # Determine reliability label
        if factuality_score >= FACTUALITY_THRESHOLDS["high_reliability"]:
            reliability_label = "high"
        elif factuality_score >= FACTUALITY_THRESHOLDS["medium_reliability"]:
            reliability_label = "medium"
        else:
            reliability_label = "low"
        
        return (
            float(factuality_score),
            reliability_label,
            float(keyword_score),
            float(credibility_score),
            float(engagement_score),
            float(ml_score)
        )
    
    def detect_factuality(self, df):
        """
        Detect factuality for all tweets in DataFrame
        
        Args:
            df: Spark DataFrame with tweet data
            
        Returns:
            DataFrame with factuality columns added
        """
        # For small datasets or when ML is enabled, use UDF approach
        # For large streaming datasets with no ML, use Spark-native approach
        
        if self.use_ml:
            # Use UDF for ML-based detection (required for model inference)
            from pyspark.sql.types import (
                DoubleType,
                StringType,
                StructField,
                StructType,
            )
            
            result_schema = StructType([
                StructField("factuality_score", DoubleType(), False),
                StructField("reliability_label", StringType(), False),
                StructField("keyword_score", DoubleType(), False),
                StructField("credibility_score", DoubleType(), False),
                StructField("engagement_score", DoubleType(), False),
                StructField("ml_score", DoubleType(), False)
            ])
            
            def factuality_udf(text, followers, retweets, favorites, replies):
                if text is None:
                    text = ""
                if followers is None:
                    followers = 0
                if retweets is None:
                    retweets = 0
                if favorites is None:
                    favorites = 0
                if replies is None:
                    replies = 0
                
                return self._calculate_factuality_score(text, followers, retweets, favorites, replies)
            
            from pyspark.sql.functions import udf
            factuality_func = udf(factuality_udf, result_schema)
            
            result_df = df.withColumn(
                "factuality_result",
                factuality_func(
                    col("tweet_text"),
                    col("followers_count"),
                    col("retweet_count"),
                    col("favorite_count"),
                    col("reply_count")
                )
            )
            
            result_df = result_df \
                .withColumn("factuality_score", col("factuality_result.factuality_score")) \
                .withColumn("reliability_label", col("factuality_result.reliability_label")) \
                .withColumn("keyword_score", col("factuality_result.keyword_score")) \
                .withColumn("credibility_score", col("factuality_result.credibility_score")) \
                .withColumn("engagement_score", col("factuality_result.engagement_score")) \
                .withColumn("ml_factuality_score", col("factuality_result.ml_score")) \
                .drop("factuality_result")
            
            return result_df
        else:
            # Spark-native computation (no Python UDFs) - faster for large datasets
            from pyspark.sql.functions import greatest, least, log10, size, split

            # Guards
            safe_word_count = F.when(col("tweet_text").isNull() | (col("tweet_text") == ""), lit(1)).otherwise(size(split(col("tweet_text"), " ")))
            followers = F.coalesce(col("followers_count"), lit(0))
            retweets = F.coalesce(col("retweet_count"), lit(0))
            favorites = F.coalesce(col("favorite_count"), lit(0))
            replies = F.coalesce(col("reply_count"), lit(0))
            total_engagement = (retweets + favorites + replies)

            # Keyword score via regex hits
            text_l = lower(col("tweet_text"))
            pos_hits = (
                text_l.rlike(r".*\bverified\b.*").cast("int") +
                text_l.rlike(r".*fact[- ]check.*").cast("int") +
                text_l.rlike(r".*\bfactual\b.*").cast("int") +
                text_l.rlike(r".*reliable source.*").cast("int") +
                text_l.rlike(r".*\bconfirmed\b.*").cast("int") +
                text_l.rlike(r".*\bofficial\b.*").cast("int") +
                text_l.rlike(r".*\bauthentic\b.*").cast("int") +
                text_l.rlike(r".*\bcredible\b.*").cast("int")
            )
            neg_hits = (
                text_l.rlike(r".*fake news.*").cast("int") +
                text_l.rlike(r".*\bmisinformation\b.*").cast("int") +
                text_l.rlike(r".*\bdisinformation\b.*").cast("int") +
                text_l.rlike(r".*\bunverified\b.*").cast("int") +
                text_l.rlike(r".*\bunconfirmed\b.*").cast("int") +
                text_l.rlike(r".*\brumor\b.*").cast("int") +
                text_l.rlike(r".*\balleged\b.*").cast("int") +
                text_l.rlike(r".*\bunsubstantiated\b.*").cast("int") +
                text_l.rlike(r".*\bhoax\b.*").cast("int")
            )
            raw_kw = (pos_hits - neg_hits) / F.greatest(safe_word_count.cast("double"), lit(1.0))
            keyword_score = least(greatest((raw_kw + lit(1.0)) / lit(2.0), lit(0.0)), lit(1.0))

            # Credibility score
            follower_score = least(log10(followers + lit(1.0)) / lit(7.0), lit(1.0))
            engagement_ratio = least((total_engagement / greatest(followers.cast("double"), lit(1.0))), lit(0.1))
            engagement_score_c = least(engagement_ratio * lit(10.0), lit(1.0))
            credibility_score = least(greatest((follower_score * lit(0.6)) + (engagement_score_c * lit(0.4)), lit(0.0)), lit(1.0))

            # Engagement quality
            favorite_reply_ratio = favorites.cast("double") / greatest(replies.cast("double"), lit(1.0))
            normalized_ratio = least(favorite_reply_ratio / lit(10.0), lit(1.0))
            volume_factor = least(log10(total_engagement + lit(1.0)) / lit(5.0), lit(1.0))
            engagement_score = (normalized_ratio * lit(0.5)) + (volume_factor * lit(0.5))

            # Final factuality (heuristic-only)
            factuality_score = (keyword_score * lit(0.4)) + (credibility_score * lit(0.3)) + (engagement_score * lit(0.3))

            high_thr = FACTUALITY_THRESHOLDS["high_reliability"]
            med_thr = FACTUALITY_THRESHOLDS["medium_reliability"]

            reliability_label = (
                when(factuality_score >= lit(high_thr), lit("high"))
                .when(factuality_score >= lit(med_thr), lit("medium"))
                .otherwise(lit("low"))
            )

            result_df = df.withColumn("keyword_score", keyword_score.cast("double")) \
                .withColumn("credibility_score", credibility_score.cast("double")) \
                .withColumn("engagement_score", engagement_score.cast("double")) \
                .withColumn("factuality_score", factuality_score.cast("double")) \
                .withColumn("reliability_label", reliability_label) \
                .withColumn("ml_factuality_score", lit(0.5))  # No ML score in heuristic mode

            return result_df
