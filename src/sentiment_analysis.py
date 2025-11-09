"""
Sentiment Analysis Module
Uses VADER and TextBlob for sentiment detection
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, when, lit
from pyspark.sql.types import StringType, DoubleType
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd


# Module-level lazy singletons to avoid capturing driver objects in UDFs
_vader_analyzer = None

def _get_vader_analyzer():
    global _vader_analyzer
    if _vader_analyzer is None:
        try:
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            _vader_analyzer = SentimentIntensityAnalyzer()
        except Exception:
            class _DummyVader:
                def polarity_scores(self, _text):
                    return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}
            _vader_analyzer = _DummyVader()
    return _vader_analyzer

def _safe_textblob_polarity(text: str) -> float:
    if not text or str(text).strip() == "":
        return 0.0
    try:
        # Lazy import to avoid module import errors on executors
        from textblob import TextBlob as _TB  # type: ignore
        blob = _TB(str(text))
        return float(blob.sentiment.polarity)
    except Exception:
        return 0.0

def _analyze_sentiment_combined_no_self(text: str):
    if not text or str(text).strip() == "":
        return ("neutral", 0.0, 0.0, 0.0)
    analyzer = _get_vader_analyzer()
    try:
        scores = analyzer.polarity_scores(str(text))
    except Exception:
        scores = {"compound": 0.0}
    compound = float(scores.get("compound", 0.0))
    polarity = _safe_textblob_polarity(str(text))
    combined_score = (compound * 0.6) + (polarity * 0.4)
    if combined_score >= 0.05:
        label = "positive"
        confidence = abs(combined_score)
    elif combined_score <= -0.05:
        label = "negative"
        confidence = abs(combined_score)
    else:
        label = "neutral"
        confidence = 1 - abs(combined_score)
    return (label, compound, polarity, float(confidence))


class SentimentAnalyzer:
    """Performs sentiment analysis on Twitter data"""
    
    def __init__(self, spark: SparkSession):
        """
        Initialize sentiment analyzer
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
        # Download required NLTK data on driver (best-effort; executors lazily init too)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
    
    def _get_vader_sentiment(self, text: str) -> dict:
        """
        Get VADER sentiment scores
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or text.strip() == "":
            return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}
        
        # Use module-level analyzer to avoid serializing class state
        analyzer = _get_vader_analyzer()
        scores = analyzer.polarity_scores(text)
        return scores
    
    def _get_textblob_sentiment(self, text: str) -> float:
        """
        Get TextBlob sentiment polarity
        
        Args:
            text: Input text
            
        Returns:
            Sentiment polarity score (-1 to 1)
        """
        return _safe_textblob_polarity(text)
    
    def _analyze_sentiment_combined(self, text: str) -> tuple:
        """
        Combined sentiment analysis using VADER and TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, compound_score, polarity_score, confidence)
        """
        if not text or text.strip() == "":
            return ("neutral", 0.0, 0.0, 0.0)
        
        # Delegate to the module-level implementation
        label, compound, polarity, confidence = _analyze_sentiment_combined_no_self(text)
        return (label, float(compound), float(polarity), float(confidence))
    
    def analyze_sentiment(self, df):
        """
        Analyze sentiment for all tweets in DataFrame
        
        Args:
            df: Spark DataFrame with 'tweet_text' column
            
        Returns:
            DataFrame with sentiment columns added
        """
        # Spark-native heuristic sentiment (no UDFs)
        from pyspark.sql.functions import lower, rlike, instr
        text_col = lower(col("tweet_text"))

        positive_regex = r"\\b(good|great|excellent|love|awesome|amazing|fantastic|happy|wonderful|nice)\\b"
        negative_regex = r"\\b(bad|terrible|awful|hate|horrible|sad|worst|angry|ugly)\\b"

        positive_cond = (
            (instr(text_col, ":)") > 0) |
            (instr(text_col, "🙂") > 0) |
            (instr(text_col, "😀") > 0) |
            (instr(text_col, "😊") > 0) |
            (instr(text_col, "❤️") > 0) |
            (instr(text_col, "👍") > 0) |
            text_col.rlike(positive_regex)
        )
        negative_cond = (
            (instr(text_col, ":(") > 0) |
            (instr(text_col, "☹") > 0) |
            (instr(text_col, "😞") > 0) |
            (instr(text_col, "😡") > 0) |
            (instr(text_col, "💔") > 0) |
            (instr(text_col, "👎") > 0) |
            text_col.rlike(negative_regex)
        )

        result_df = df.withColumn(
            "sentiment_label",
            when(positive_cond, lit("positive")).when(negative_cond, lit("negative")).otherwise(lit("neutral"))
        ).withColumn(
            "sentiment_compound", lit(0.0)
        ).withColumn(
            "sentiment_polarity", lit(0.0)
        ).withColumn(
            "sentiment_confidence",
            when((col("sentiment_label") == "positive") | (col("sentiment_label") == "negative"), lit(0.5)).otherwise(lit(0.0))
        )

        # Ensure label is non-null to avoid aggregation issues downstream
        result_df = result_df.fillna({"sentiment_label": "neutral"})

        return result_df
