"""
Configuration settings for Twitter Data Analysis Project
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Spark configuration
SPARK_CONFIG = {
    "app_name": "TwitterMultimodalAnalysis",
    "master": "local[*]",  # Use 'local[*]' for local, 'yarn' for cluster
    "spark.executor.memory": "2g",
    "spark.driver.memory": "2g",
    "spark.sql.shuffle.partitions": "200",
    "spark.default.parallelism": "8",
    # Java 21+ compatibility settings
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true"
}

# Analysis settings
SENTIMENT_THRESHOLDS = {
    "positive": 0.05,
    "negative": -0.05
}

FACTUALITY_THRESHOLDS = {
    "high_reliability": 0.7,
    "medium_reliability": 0.4,
    "low_reliability": 0.0
}

# Keywords for factuality detection
FACT_CHECK_KEYWORDS = [
    "verified", "fact-check", "debunk", "misinformation",
    "disinformation", "fake news", "factual", "reliable source",
    "unverified", "unconfirmed", "rumor", "alleged"
]

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
