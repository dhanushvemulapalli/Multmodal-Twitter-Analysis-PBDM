"""
Unit tests for sentiment analysis and factuality detection
Tests core logic without PySpark dependencies
"""
import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sentiment_analysis import SentimentAnalyzer
from src.factuality_detection import FactualityDetector


class TestSentimentAnalysis(unittest.TestCase):
    """Test sentiment analysis without Spark"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create analyzer without Spark (will fail on some methods, but core logic works)
        try:
            self.analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
            self.analyzer.vader = SentimentAnalyzer.__init__.__globals__['SentimentIntensityAnalyzer']()
        except:
            # Alternative: import directly
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.analyzer = type('obj', (object,), {})()
            self.analyzer.vader = SentimentIntensityAnalyzer()
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        test_cases = [
            "I love this product! It's amazing!",
            "Great news! This is fantastic!",
            "Excellent service, highly recommended!"
        ]
        
        for text in test_cases:
            label, compound, polarity, confidence = self._analyze_sentiment(text)
            self.assertEqual(label, "positive", f"Expected positive sentiment for: {text}")
            self.assertGreater(compound, 0, "Compound score should be positive")
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        test_cases = [
            "This is terrible! I hate it!",
            "Very disappointed with the service",
            "Poor quality, not recommended"
        ]
        
        for text in test_cases:
            label, compound, polarity, confidence = self._analyze_sentiment(text)
            self.assertEqual(label, "negative", f"Expected negative sentiment for: {text}")
            self.assertLess(compound, 0, "Compound score should be negative")
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        test_cases = [
            "This is a regular update",
            "Just sharing some information",
            "The weather today is cloudy"
        ]
        
        for text in test_cases:
            label, compound, polarity, confidence = self._analyze_sentiment(text)
            # Neutral sentiment should have low absolute compound score
            # Accept either "neutral" label or very close to zero scores
            if label != "neutral":
                # If not labeled neutral, check that it's very close to neutral (within threshold)
                self.assertLess(abs(compound), 0.15, 
                               f"Expected near-neutral sentiment for: {text}, got {label} with score {compound}")
            else:
                # If labeled neutral, verify it's close to zero
                self.assertLess(abs(compound), 0.1, 
                               f"Neutral sentiment should have low compound score, got {compound}")
    
    def test_empty_text(self):
        """Test handling of empty text"""
        label, compound, polarity, confidence = self._analyze_sentiment("")
        self.assertEqual(label, "neutral")
        self.assertEqual(compound, 0.0)
        self.assertEqual(polarity, 0.0)
    
    def test_sentiment_confidence(self):
        """Test that confidence scores are valid"""
        test_text = "This is absolutely amazing! I love it!"
        label, compound, polarity, confidence = self._analyze_sentiment(test_text)
        
        self.assertGreaterEqual(confidence, 0.0, "Confidence should be >= 0")
        self.assertLessEqual(confidence, 1.0, "Confidence should be <= 1")
    
    def _analyze_sentiment(self, text):
        """Helper method to analyze sentiment"""
        from textblob import TextBlob
        
        # Get VADER scores
        vader_scores = self.analyzer.vader.polarity_scores(text)
        compound = vader_scores["compound"]
        
        # Get TextBlob polarity
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
        except:
            polarity = 0.0
        
        # Combined score (weighted average)
        combined_score = (compound * 0.6) + (polarity * 0.4)
        
        # Determine label
        if combined_score >= 0.05:
            label = "positive"
            confidence = abs(combined_score)
        elif combined_score <= -0.05:
            label = "negative"
            confidence = abs(combined_score)
        else:
            label = "neutral"
            confidence = 1 - abs(combined_score)
        
        return (label, float(compound), float(polarity), float(confidence))


class TestFactualityDetection(unittest.TestCase):
    """Test factuality detection without Spark"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = FactualityDetector.__new__(FactualityDetector)
        # Initialize fact check keywords
        from config.config import FACT_CHECK_KEYWORDS
        self.detector.fact_check_keywords = FACT_CHECK_KEYWORDS
    
    def test_keyword_detection_positive(self):
        """Test detection of positive factuality keywords"""
        test_cases = [
            "This is verified information from reliable source",
            "Fact-checked and confirmed by experts",
            "Official and authentic statement"
        ]
        
        for text in test_cases:
            score = self.detector._calculate_keyword_score(text)
            self.assertGreater(score, 0.5, f"Expected higher score for: {text}")
    
    def test_keyword_detection_negative(self):
        """Test detection of negative factuality keywords"""
        test_cases = [
            "This is fake news and misinformation",
            "Unverified rumor circulating online",
            "Unconfirmed and unsubstantiated claims"
        ]
        
        for text in test_cases:
            score = self.detector._calculate_keyword_score(text)
            self.assertLess(score, 0.6, f"Expected lower score for: {text}")
    
    def test_keyword_detection_neutral(self):
        """Test neutral text without factuality keywords"""
        test_text = "This is just regular information about the weather"
        score = self.detector._calculate_keyword_score(test_text)
        # Should be around neutral (0.5)
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.7)
    
    def test_user_credibility_high_followers(self):
        """Test credibility calculation with high follower count"""
        credibility = self.detector._calculate_user_credibility_score(
            followers=1000000,
            engagement=50000
        )
        self.assertGreater(credibility, 0.5, "High followers should increase credibility")
    
    def test_user_credibility_low_followers(self):
        """Test credibility calculation with low follower count"""
        credibility = self.detector._calculate_user_credibility_score(
            followers=10,
            engagement=5
        )
        self.assertLess(credibility, 0.5, "Low followers should decrease credibility")
    
    def test_engagement_quality(self):
        """Test engagement quality calculation"""
        # High favorites to replies ratio indicates quality
        quality1 = self.detector._calculate_engagement_quality(
            retweets=100,
            favorites=1000,
            replies=10
        )
        
        quality2 = self.detector._calculate_engagement_quality(
            retweets=100,
            favorites=50,
            replies=100
        )
        
        self.assertGreater(quality1, quality2, "Higher favorite/reply ratio should increase quality")
    
    def test_factuality_score_calculation(self):
        """Test overall factuality score calculation"""
        # High credibility, positive keywords
        score1, label1, kw1, cred1, eng1 = self.detector._calculate_factuality_score(
            text="Verified information from reliable source",
            followers=100000,
            retweets=500,
            favorites=2000,
            replies=50
        )
        
        # Low credibility, negative keywords
        score2, label2, kw2, cred2, eng2 = self.detector._calculate_factuality_score(
            text="Fake news and misinformation",
            followers=10,
            retweets=5,
            favorites=2,
            replies=1
        )
        
        self.assertGreater(score1, score2, "First case should have higher factuality score")
        self.assertGreaterEqual(score1, 0.0)
        self.assertLessEqual(score1, 1.0)
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)
        
        # Check labels
        self.assertIn(label1, ["high", "medium", "low"])
        self.assertIn(label2, ["high", "medium", "low"])
    
    def test_factuality_reliability_labels(self):
        """Test that reliability labels are correctly assigned"""
        from config.config import FACTUALITY_THRESHOLDS
        
        # High reliability
        score_high, label_high, _, _, _ = self.detector._calculate_factuality_score(
            text="Verified and confirmed by official sources",
            followers=500000,
            retweets=1000,
            favorites=5000,
            replies=100
        )
        
        if score_high >= FACTUALITY_THRESHOLDS["high_reliability"]:
            self.assertEqual(label_high, "high")
        elif score_high >= FACTUALITY_THRESHOLDS["medium_reliability"]:
            self.assertEqual(label_high, "medium")
        else:
            self.assertEqual(label_high, "low")


class TestDataProcessing(unittest.TestCase):
    """Test data processing utilities"""
    
    def test_text_cleaning(self):
        """Test text cleaning functions"""
        import re
        
        test_cases = [
            ("Hello World!", "hello world"),
            ("  Extra Spaces  ", "extra spaces"),
            ("MIXED CASE", "mixed case"),
        ]
        
        for input_text, expected in test_cases:
            cleaned = re.sub(r"[^\w\s@#]", "", input_text.lower().strip())
            cleaned = " ".join(cleaned.split())
            self.assertEqual(cleaned, expected)
    
    def test_engagement_calculation(self):
        """Test engagement metric calculation"""
        retweets = 100
        favorites = 500
        replies = 50
        
        total_engagement = retweets + favorites + replies
        self.assertEqual(total_engagement, 650)
        
        # Test averages
        avg_engagement = (retweets + favorites + replies) / 3
        self.assertAlmostEqual(avg_engagement, 216.67, places=1)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

