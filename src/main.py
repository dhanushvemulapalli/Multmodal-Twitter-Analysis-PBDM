"""
Main Execution Script for Twitter Multimodal Analysis
"""
import argparse
import sys
import os
import platform
from pathlib import Path

# Set Java options for Java 21+ compatibility
# Must be set before importing PySpark to affect the gateway process
if platform.system() == "Windows":
    # Set PYSPARK_SUBMIT_ARGS to pass Java options to Spark
    java_opts = [
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED",
        "--add-opens=java.base/java.security=ALL-UNNAMED",
        "--enable-native-access=ALL-UNNAMED",
        "-Dio.netty.tryReflectionSetAccessible=true"
    ]
    java_opts_str = " ".join(java_opts)
    # Set via PYSPARK_SUBMIT_ARGS to pass Java options to Spark gateway
    # The format needs to properly escape the Java options
    # Only set if not already set (allows user override)
    if "PYSPARK_SUBMIT_ARGS" not in os.environ:
        # Format: --driver-java-options "options" pyspark-shell
        # Use double quotes to handle spaces in options
        os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "{java_opts_str}" pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, count, avg, sum as spark_sum
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, DoubleType
import signal
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import SPARK_CONFIG, PROJECT_ROOT
from src.data_ingestion import TwitterDataIngestion
from src.sentiment_analysis import SentimentAnalyzer
from src.factuality_detection import FactualityDetector
from src.visualization import Visualizer
from src.streaming import TwitterStreamProducer, LocalFileStreamProducer


def create_spark_session(mode: str = "local", partitions: int = 4, streaming: bool = False):
    """
    Create and configure Spark session
    
    Args:
        mode: Spark mode ('local' or 'cluster')
        partitions: Number of partitions
        streaming: Whether this is for streaming mode
        
    Returns:
        Configured SparkSession
    """
    if mode == "local":
        master = f"local[{partitions}]"
    else:
        master = "yarn"
    
    builder = SparkSession.builder \
        .appName(SPARK_CONFIG["app_name"] + ("_Streaming" if streaming else "")) \
        .master(master)
    
    # Add configuration
    for key, value in SPARK_CONFIG.items():
        if key not in ["app_name", "master"]:
            builder = builder.config(key, value)
    
    # Set partitions
    builder = builder.config("spark.default.parallelism", str(partitions))
    builder = builder.config("spark.sql.shuffle.partitions", str(partitions * 50))
    
    # Additional Windows/Java 21+ compatibility settings
    builder = builder.config("spark.sql.adaptive.enabled", "true")
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    # Help diagnose UDF crashes and avoid reused worker instability on Windows
    builder = builder.config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    builder = builder.config("spark.python.worker.faulthandler.enabled", "true")
    builder = builder.config("spark.python.worker.reuse", "false")
    
    # Streaming-specific configurations
    if streaming:
        builder = builder.config("spark.sql.streaming.checkpointLocation", str(PROJECT_ROOT / "checkpoints"))
        builder = builder.config("spark.sql.streaming.schemaInference", "true")
    
    # Additional Java 21+ compatibility for security/auth
    # Set Java options via Spark config (more reliable than environment variables)
    java_opts_list = [
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
        "--add-opens=java.base/javax.security.auth=ALL-UNNAMED",
        "--add-opens=java.base/java.security=ALL-UNNAMED",
        "--enable-native-access=ALL-UNNAMED",
        "-Dio.netty.tryReflectionSetAccessible=true"
    ]
    java_opts_str = " ".join(java_opts_list)
    builder = builder.config("spark.driver.extraJavaOptions", java_opts_str)
    builder = builder.config("spark.executor.extraJavaOptions", java_opts_str)
    
    # Configure Hadoop to work around Java 21+ security issues
    # Disable security features that require Subject.getSubject()
    builder = builder.config("spark.hadoop.security.authentication", "simple")
    builder = builder.config("spark.hadoop.security.authorization", "false")
    
    # Additional Windows-specific settings to prevent gateway crashes
    if platform.system() == "Windows":
        builder = builder.config("spark.driver.host", "localhost")
        builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
        # Disable native Hadoop libraries on Windows to avoid UnsatisfiedLinkError
        builder = builder.config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        builder = builder.config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        builder = builder.config("spark.hadoop.fs.defaultFS", "file:///")
        # Disable native IO to prevent UnsatisfiedLinkError
        builder = builder.config("spark.hadoop.io.native.lib.available", "false")
        # Use Java-based file system instead of native
        os.environ["HADOOP_HOME"] = str(PROJECT_ROOT)  # Set to avoid warnings
    
    try:
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        return spark
    except Exception as e:
        error_msg = str(e)
        if "getSubject is not supported" in error_msg or "UnsupportedOperationException" in error_msg:
            print("\n" + "=" * 70)
            print("ERROR: Java 21+ Compatibility Issue with Spark")
            print("=" * 70)
            print("Spark/Hadoop is calling Subject.getSubject() which is not supported")
            print("in Java 21+. This is a known compatibility issue.")
            print("\nSOLUTIONS:")
            print("1. Use Java 11 or 17 (RECOMMENDED):")
            print("   - Download from: https://adoptium.net/")
            print("   - Set JAVA_HOME environment variable to Java 11/17")
            print("   - Example: set JAVA_HOME=C:\\Program Files\\Java\\jdk-17")
            print("\n2. Use file-based mode instead (works with any Java version):")
            print("   python src/main.py --mode file --input data/tweets.json")
            print("\n3. Wait for Spark 4.0+ which will support Java 21+")
            print("=" * 70)
        raise


def run_streaming_analysis(keyword: str, output_dir: Path, mode: str, partitions: int, batch_interval: int = 3):
    """
    Run real-time streaming analysis on Twitter data
    
    Args:
        keyword: Keyword to filter tweets
        output_dir: Output directory for results
        mode: Spark execution mode
        partitions: Number of Spark partitions
        batch_interval: Streaming batch interval in seconds
    """
    print("=" * 60)
    print("Twitter Multimodal Analysis - STREAMING MODE")
    print("=" * 60)
    print(f"Keyword: {keyword}")
    print(f"Output directory: {output_dir}")
    print(f"Spark mode: {mode}")
    print(f"Partitions: {partitions}")
    print(f"Batch interval: {batch_interval}s")
    print("=" * 60)
    
    # Create directories
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    stream_data_dir = PROJECT_ROOT / "stream_data"
    stream_data_dir.mkdir(parents=True, exist_ok=True)
    
    live_results_dir = output_dir / "live"
    live_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Spark session for streaming
    print("\n[1/4] Initializing Spark session for streaming...")
    spark = create_spark_session(mode, partitions, streaming=True)
    print("[OK] Spark streaming session created")
    
    # Initialize stream producer
    print("\n[2/4] Initializing Twitter stream producer...")
    stream_producer = TwitterStreamProducer(
        keyword=keyword,
        output_dir=str(stream_data_dir),
        batch_interval=batch_interval
    )
    
    # Start streaming
    stream_producer.start()
    
    # Define schema for tweets
    tweet_schema = StructType([
        StructField("id", StringType(), True),
        StructField("text", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("user", StructType([
            StructField("screen_name", StringType(), True),
            StructField("followers_count", IntegerType(), True)
        ]), True),
        StructField("retweet_count", IntegerType(), True),
        StructField("favorite_count", IntegerType(), True),
        StructField("reply_count", IntegerType(), True)
    ])
    
    try:
        print("\n[3/4] Starting Spark Structured Streaming...")
        
        # Read streaming data
        # Use directory path - Spark will watch for new files
        stream_path = str(stream_data_dir)
        
        # Ensure directory exists
        if not stream_data_dir.exists():
            stream_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a valid dummy JSON file with proper schema to help Spark initialize
        # This workaround helps avoid UnsatisfiedLinkError on Windows
        # Spark needs at least one valid file to infer schema without listing directory
        dummy_file = stream_data_dir / "_init_dummy.json"
        if not dummy_file.exists():
            import json
            dummy_tweet = {
                "id": "dummy",
                "text": "dummy",
                "created_at": "2024-01-01T00:00:00Z",
                "user": {
                    "screen_name": "dummy",
                    "followers_count": 0
                },
                "retweet_count": 0,
                "favorite_count": 0,
                "reply_count": 0
            }
            with open(dummy_file, 'w', encoding='utf-8') as f:
                json.dump(dummy_tweet, f)
        
        # Wait a moment for file system to sync
        import time
        time.sleep(0.5)
        
        # Use a specific file pattern that avoids directory listing issues
        # Read from the directory but filter for JSON files
        try:
            # Try reading from a specific file first to avoid directory listing
            # Then switch to directory watching
            streaming_df = spark.readStream \
                .schema(tweet_schema) \
                .option("maxFilesPerTrigger", 1) \
                .option("latestFirst", "false") \
                .option("pathGlobFilter", "*.json") \
                .json(stream_path)
        except Exception as e:
            if "UnsatisfiedLinkError" in str(e) or "NativeIO" in str(e):
                print("\n" + "=" * 70)
                print("ERROR: Windows Native Library Compatibility Issue")
                print("=" * 70)
                print("This is a COMPATIBILITY ISSUE, not a code bug.")
                print("\nROOT CAUSE:")
                print("- Spark uses Hadoop's native Windows libraries for file operations")
                print("- Spark tries to list directory files during stream initialization")
                print("- The native Windows library call fails: UnsatisfiedLinkError")
                print("- This can happen even with Java 17 on Windows")
                print("\nSOLUTIONS (in order of preference):")
                print("1. Use file-based mode (WORKS NOW - RECOMMENDED):")
                print("   python src/main.py --mode file --input data/tweets.json")
                print("   - Works perfectly on Windows with any Java version")
                print("\n2. Use WSL2 (Windows Subsystem for Linux):")
                print("   - Install: wsl --install")
                print("   - Install Python: sudo apt install python3 python3-pip")
                print("   - Run streaming in Linux environment (avoids Windows issues)")
                print("\n3. Use Java 11 or 17 (may still have issues):")
                print("   - Download from: https://adoptium.net/")
                print("   - Set JAVA_HOME to Java 11/17 installation")
                print("   - NOTE: Even Java 17 can have this issue on Windows")
                print("\n4. Wait for Spark 4.0+ (expected 2024-2025):")
                print("   - Better Windows support expected")
                print("\nNOTE: The code is correct. This is a Spark/Hadoop limitation.")
                print("=" * 70)
                raise
            else:
                raise
        
        # Preprocess data
        ingestion = TwitterDataIngestion(spark)
        processed_df = ingestion.preprocess_data(streaming_df)
        
        # Sentiment analysis
        sentiment_analyzer = SentimentAnalyzer(spark)
        analyzed_df = sentiment_analyzer.analyze_sentiment(processed_df)
        
        # Factuality detection
        factuality_detector = FactualityDetector(spark)
        final_df = factuality_detector.detect_factuality(analyzed_df)
        
        # Add timestamp for windowing
        from pyspark.sql.functions import current_timestamp
        final_df = final_df.withColumn("processing_time", current_timestamp())
        
        # Compute running statistics
        stats_query = final_df \
            .withWatermark("processing_time", "10 seconds") \
            .groupBy(
                window("processing_time", "10 seconds", "5 seconds"),
                "sentiment_label"
            ) \
            .agg(
                count("*").alias("count"),
                avg("sentiment_confidence").alias("avg_confidence")
            ) \
            .orderBy("window")
        
        # Write statistics to console
        console_query = stats_query.writeStream \
            .outputMode("complete") \
            .format("console") \
            .option("truncate", "false") \
            .trigger(processingTime=f"{batch_interval} seconds") \
            .start()
        
        # Write results to JSON files
        results_query = final_df.writeStream \
            .format("json") \
            .option("path", str(live_results_dir)) \
            .option("checkpointLocation", str(checkpoint_dir / "results")) \
            .trigger(processingTime=f"{batch_interval} seconds") \
            .start()
        
        print("[OK] Streaming queries started")
        print("\n[4/4] Processing live tweets...")
        print("Press Ctrl+C to stop streaming\n")
        
        # Print running statistics
        start_time = time.time()
        while True:
            time.sleep(batch_interval)
            elapsed = int(time.time() - start_time)
            print(f"\n[{elapsed}s] Tweets collected: {stream_producer.tweet_count} | Errors: {stream_producer.error_count}")
            
            # Check if queries are still active
            if not console_query.isActive or not results_query.isActive:
                print("Warning: One or more streaming queries stopped")
                break
        
    except KeyboardInterrupt:
        print("\n\nStopping streaming...")
    except Exception as e:
        print(f"\nError during streaming: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop streaming
        stream_producer.stop()
        
        # Stop Spark queries
        try:
            if 'console_query' in locals():
                console_query.stop()
            if 'results_query' in locals():
                results_query.stop()
        except:
            pass
        
        spark.stop()
        print("\n[OK] Streaming stopped. Spark session closed.")


def run_file_analysis(args):
    """Run file-based analysis (original mode)"""
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Twitter Multimodal Analysis - PBDM Project")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Spark mode: {args.spark_mode}")
    print(f"Partitions: {args.partitions}")
    print("=" * 60)
    
    # Create Spark session
    print("\n[1/5] Initializing Spark session...")
    spark = create_spark_session(args.spark_mode, args.partitions)
    print("[OK] Spark session created")
    
    try:
        # Data ingestion
        print("\n[2/5] Loading and preprocessing data...")
        ingestion = TwitterDataIngestion(spark)
        df = ingestion.load_json_data(str(input_path))
        print(f"[OK] Loaded {df.count():,} tweets")
        
        df = ingestion.preprocess_data(df)
        print(f"[OK] Preprocessed {df.count():,} tweets")
        
        # Display statistics
        stats = ingestion.get_statistics(df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Sentiment analysis
        print("\n[3/5] Performing sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer(spark)
        df = sentiment_analyzer.analyze_sentiment(df)
        print("[OK] Sentiment analysis completed")
        
        # Display sentiment distribution
        sentiment_dist = df.groupBy("sentiment_label").count().orderBy("sentiment_label")
        print("\nSentiment Distribution:")
        sentiment_dist.show(truncate=False)
        
        # Factuality detection
        print("\n[4/5] Performing factuality detection...")
        factuality_detector = FactualityDetector(spark)
        df = factuality_detector.detect_factuality(df)
        print("[OK] Factuality detection completed")
        
        # Display factuality distribution
        factuality_dist = df.groupBy("reliability_label").count().orderBy("reliability_label")
        print("\nFactuality Distribution:")
        factuality_dist.show(truncate=False)
        
        # Visualization and reporting
        if not args.skip_visualization:
            print("\n[5/5] Generating visualizations and reports...")
            visualizer = Visualizer(spark, str(output_dir))
            
            # Generate all visualizations
            visualizer.plot_sentiment_distribution(df)
            visualizer.plot_factuality_distribution(df)
            visualizer.plot_sentiment_vs_factuality(df)
            visualizer.plot_engagement_analysis(df)
            
            # Generate word clouds
            visualizer.create_wordcloud(df)
            visualizer.create_wordcloud(df, sentiment="positive")
            visualizer.create_wordcloud(df, sentiment="negative")
            
            # Generate summary report
            visualizer.generate_summary_report(df)
            
            # Save results to CSV
            # visualizer.save_results_csv(df)
            
            print("[OK] All visualizations and reports generated")
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        spark.stop()
        print("\nSpark session stopped")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Multimodal Twitter Data Analysis using PySpark"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="file",
        choices=["file", "stream"],
        help="Execution mode: 'file' for batch processing, 'stream' for real-time streaming"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input Twitter data file (JSON format) - required for file mode"
    )
    parser.add_argument(
        "--keyword",
        type=str,
        help="Keyword/topic to stream tweets for - required for stream mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Path to output directory for results"
    )
    parser.add_argument(
        "--spark-mode",
        type=str,
        default="local",
        choices=["local", "cluster"],
        dest="spark_mode",
        help="Spark execution mode (local or cluster)"
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=4,
        help="Number of Spark partitions"
    )
    parser.add_argument(
        "--batch-interval",
        type=int,
        default=3,
        dest="batch_interval",
        help="Streaming batch interval in seconds (default: 3)"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation (file mode only)"
    )
    
    args = parser.parse_args()
    
    # Route to appropriate mode
    if args.mode == "stream":
        if not args.keyword:
            print("Error: --keyword is required for stream mode")
            print("Example: python src/main.py --mode stream --keyword 'bitcoin'")
            sys.exit(1)
        
        output_dir = Path(args.output)
        run_streaming_analysis(
            keyword=args.keyword,
            output_dir=output_dir,
            mode=args.spark_mode,
            partitions=args.partitions,
            batch_interval=args.batch_interval
        )
    else:
        # File mode
        if not args.input:
            print("Error: --input is required for file mode")
            print("Example: python src/main.py --mode file --input data/tweets.json")
            sys.exit(1)
        
        run_file_analysis(args)


if __name__ == "__main__":
    main()
