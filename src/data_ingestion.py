"""
Twitter Data Ingestion and Preprocessing Module
Handles loading and preprocessing Twitter data using PySpark
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, regexp_replace, lower, trim,
    split, size, lit
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType
from pyspark.sql import functions as F
import json
from typing import Optional


class TwitterDataIngestion:
    """Handles Twitter data ingestion and preprocessing"""
    
    def __init__(self, spark: SparkSession):
        """
        Initialize the data ingestion module
        
        Args:
            spark: SparkSession instance
        """
        self.spark = spark
    
    def load_json_data(self, file_path: str) -> 'DataFrame':
        """
        Load Twitter data from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Spark DataFrame
        """
        try:
            # Try loading as JSON
            df = self.spark.read.json(file_path)
            return df
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            # Try loading as JSON Lines (one JSON object per line)
            try:
                df = self.spark.read.option("multiLine", "false").json(file_path)
                return df
            except Exception as e2:
                raise Exception(f"Failed to load data: {e2}")
    
    def preprocess_data(self, df) -> 'DataFrame':
        """
        Preprocess Twitter data
        
        Args:
            df: Input Spark DataFrame
            
        Returns:
            Preprocessed Spark DataFrame
        """
        # Select and rename columns
        processed_df = df.select(
            col("id").alias("tweet_id"),
            col("text").alias("tweet_text"),
            col("created_at").alias("created_at"),
            col("user.screen_name").alias("username"),
            when(col("user.followers_count").isNotNull(), col("user.followers_count"))
                .otherwise(lit(0)).alias("followers_count"),
            when(col("retweet_count").isNotNull(), col("retweet_count"))
                .otherwise(lit(0)).alias("retweet_count"),
            when(col("favorite_count").isNotNull(), col("favorite_count"))
                .otherwise(lit(0)).alias("favorite_count"),
            when(col("reply_count").isNotNull(), col("reply_count"))
                .otherwise(lit(0)).alias("reply_count")
        )
        
        # Clean text
        processed_df = processed_df.withColumn(
            "tweet_text",
            trim(lower(regexp_replace(
                col("tweet_text"),
                r"[^\w\s@#]|http\S+|www\.\S+",
                ""
            )))
        )
        
        # Remove empty tweets
        processed_df = processed_df.filter(
            col("tweet_text").isNotNull() & 
            (col("tweet_text") != "")
        )
        
        # Add text statistics
        processed_df = processed_df.withColumn(
            "text_length",
            F.length(col("tweet_text"))
        ).withColumn(
            "word_count",
            size(split(col("tweet_text"), " "))
        )
        
        # Calculate engagement metrics
        processed_df = processed_df.withColumn(
            "total_engagement",
            col("retweet_count") + col("favorite_count") + col("reply_count")
        )
        
        # Add hashtag and mention counts (simplified)
        processed_df = processed_df.withColumn(
            "hashtag_count",
            size(split(col("tweet_text"), "#")) - 1
        ).withColumn(
            "mention_count",
            size(split(col("tweet_text"), "@")) - 1
        )
        
        return processed_df
    
    def get_statistics(self, df) -> dict:
        """
        Get basic statistics about the dataset
        
        Args:
            df: Spark DataFrame
            
        Returns:
            Dictionary with statistics
        """
        from pyspark.sql.functions import avg, sum as spark_sum
        
        stats = {
            "total_tweets": df.count(),
            "unique_users": df.select("username").distinct().count(),
            "avg_text_length": df.agg(avg("text_length")).collect()[0][0],
            "avg_word_count": df.agg(avg("word_count")).collect()[0][0],
            "total_engagement": df.agg(spark_sum("total_engagement")).collect()[0][0],
            "avg_retweets": df.agg(avg("retweet_count")).collect()[0][0],
            "avg_favorites": df.agg(avg("favorite_count")).collect()[0][0]
        }
        return stats
    
    def save_processed_data(self, df, output_path: str, format: str = "parquet"):
        """
        Save processed data
        
        Args:
            df: Spark DataFrame to save
            output_path: Output file path
            format: Output format (parquet, json, csv)
        """
        if format == "parquet":
            df.write.mode("overwrite").parquet(output_path)
        elif format == "json":
            df.write.mode("overwrite").json(output_path)
        elif format == "csv":
            df.write.mode("overwrite").option("header", "true").csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")


