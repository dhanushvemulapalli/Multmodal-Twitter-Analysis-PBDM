# Multimodal Twitter Data Analysis - PBDM Project

This project performs comprehensive analysis of Twitter data using PySpark to detect sentiment and factuality. It leverages big data processing capabilities to analyze large-scale Twitter datasets efficiently.

## Features

- **Large-scale Data Processing**: Uses PySpark for distributed processing of Twitter data
- **Sentiment Analysis**: Detects sentiment (positive, negative, neutral) in tweets
- **Factuality Detection**: Assesses the factuality/reliability of tweets
- **Visualization**: Interactive visualizations of analysis results
- **Multimodal Analysis**: Processes text, metadata, and engagement metrics

## Project Structure

```
├── src/
│   ├── data_ingestion.py      # Twitter data ingestion and preprocessing
│   ├── sentiment_analysis.py  # Sentiment detection module
│   ├── factuality_detection.py # Factuality assessment module
│   ├── visualization.py       # Visualization and reporting
│   └── main.py                # Main execution script
├── config/
│   └── config.py              # Configuration settings
├── scripts/
│   ├── generate_sample_data.py # Generate test data
│   ├── setup.sh               # Setup script (Linux/Mac)
│   └── setup.bat              # Setup script (Windows)
├── notebooks/                 # Jupyter notebooks for exploratory analysis
├── data/                      # Data directory (gitignored)
├── results/                   # Analysis results and visualizations
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

1. **Generate sample data** (for testing):
   ```bash
   python scripts/generate_sample_data.py --num-tweets 100 --output data/sample_tweets.json
   ```

2. **Run the analysis**:
   ```bash
   python src/main.py --input data/sample_tweets.json --output results/
   ```

3. **View results**: Check the `results/` directory for generated visualizations and reports.

## Setup

### Prerequisites

- Python 3.8+
- Java 8, 11, or 17 (required for PySpark)
  - **Note**: Java 21+ works but requires additional configuration (handled automatically in this project)
- Apache Spark 3.5.0 (installed via PySpark package)

### Installation

#### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
scripts\setup.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

#### Option 2: Manual Setup

1. Install Java (if not already installed):
   - **Windows**: Download from https://www.oracle.com/java/technologies/downloads/
   - **Linux**: `sudo apt-get install openjdk-11-jdk` or `sudo yum install java-11-openjdk`
   - **Mac**: `brew install openjdk@11`
   
   Verify installation:
   ```bash
   java -version
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```bash
   python -m nltk.downloader vader_lexicon punkt stopwords
   ```

5. (Optional) Set up Twitter API credentials (create `.env` file):
   ```
   TWITTER_API_KEY=your_api_key
   TWITTER_API_SECRET=your_api_secret
   TWITTER_ACCESS_TOKEN=your_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
   ```
   
   Note: API credentials are optional if you're using pre-collected data.

### Troubleshooting

#### Java Compatibility Issues (Java 21+)

**Streaming Mode on Windows with Java 21+:**

If you encounter `UnsatisfiedLinkError: 'boolean org.apache.hadoop.io.nativeio.NativeIO$Windows.access0'` when using streaming mode on Windows with Java 21+, this is a **known compatibility issue** between Spark/Hadoop and Java 21+ on Windows.

**This is NOT a code bug** - it's a Spark/Hadoop limitation.

**Solutions:**
1. **Use Java 11 or 17** (RECOMMENDED):
   - Download from: https://adoptium.net/
   - Set `JAVA_HOME` environment variable to Java 11/17 installation
   - Example: `set JAVA_HOME=C:\Program Files\Java\jdk-17`

2. **Use file-based mode** (works with any Java version):
   ```bash
   python src/main.py --mode file --input data/tweets.json
   ```

3. **Use WSL2** (Windows Subsystem for Linux):
   - Run streaming in Linux environment to avoid Windows native library issues

4. **Wait for Spark 4.0+**: Better Windows and Java 21+ support expected in 2024-2025

**File-based mode works perfectly** with Java 21+ - only streaming mode has this limitation on Windows.

**Other Java Issues:**
- If you encounter `java.lang.UnsupportedOperationException: getSubject is not supported`, this project automatically handles Java 21+ compatibility. However, if issues persist, use Java 11 or 17.

**Windows Hadoop Warning:**
- The `HADOOP_HOME` warning is harmless for local development
- To suppress it, download winutils.exe and set `HADOOP_HOME` environment variable

#### Memory Issues

If you encounter out-of-memory errors:

1. Reduce Spark memory settings in `config/config.py`:
   ```python
   "spark.executor.memory": "1g",
   "spark.driver.memory": "1g"
   ```

2. Reduce partition count:
   ```bash
   python src/main.py --input data/tweets.json --output results/ --partitions 2
   ```

## Usage

### File-Based Analysis (Batch Mode)

**Basic usage with sample data:**
```bash
python src/main.py --mode file --input data/sample_tweets.json --output results/
```

**With custom configuration:**
```bash
python src/main.py --mode file --input data/tweets_sample.json --output results/ --spark-mode local --partitions 4
```

**Skip visualization (faster for large datasets):**
```bash
python src/main.py --mode file --input data/tweets.json --output results/ --skip-visualization
```

### Real-Time Streaming Mode

The project now supports real-time streaming of tweets from Twitter API v2. This mode connects to Twitter's filtered stream endpoint and processes tweets in real-time using Spark Structured Streaming.

**Prerequisites for Streaming:**
1. Twitter API v2 credentials (Bearer Token or API keys)
2. Create a `.env` file in the project root with:
   ```
   TWITTER_BEARER_TOKEN=your_bearer_token
   # OR
   TWITTER_API_KEY=your_api_key
   TWITTER_API_SECRET=your_api_secret
   TWITTER_ACCESS_TOKEN=your_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
   ```

**Basic streaming usage:**
```bash
python src/main.py --mode stream --keyword "bitcoin"
```

**Streaming with custom configuration:**
```bash
python src/main.py --mode stream --keyword "elections" --output results/ --batch-interval 5 --partitions 4
```

**Streaming features:**
- Real-time tweet collection from Twitter API v2 filtered stream
- Mini-batch processing (configurable interval: 1-5 seconds)
- Live sentiment and factuality analysis
- Running statistics displayed in terminal
- Results written to `results/live/` directory
- Automatic checkpointing for fault tolerance
- Fallback to local file streaming if API unavailable

**Streaming Command Line Options:**
- `--mode stream`: Enable streaming mode
- `--keyword`: Keyword/topic to filter tweets (required for stream mode)
- `--output`: Output directory (default: `results/`)
- `--batch-interval`: Streaming batch interval in seconds (default: 3)
- `--spark-mode`: Spark execution mode ('local' or 'cluster', default: 'local')
- `--partitions`: Number of Spark partitions (default: 4)

**Example streaming commands:**
```bash
# Stream tweets about bitcoin
python src/main.py --mode stream --keyword "bitcoin"

# Stream with 5-second batches
python src/main.py --mode stream --keyword "elections" --batch-interval 5

# Stream with custom output directory
python src/main.py --mode stream --keyword "python" --output results/live_analysis/
```

**Streaming Output:**
- Console: Real-time statistics every batch interval showing sentiment distribution
- Files: Processed tweets written to `results/live/` in JSON format
- Checkpoints: Stored in `checkpoints/` directory for fault tolerance

**Stopping Streaming:**
Press `Ctrl+C` to gracefully stop the streaming pipeline. All checkpoints are saved automatically.

### Command Line Options Summary

**File Mode:**
- `--mode file`: Batch processing mode (default)
- `--input`: Path to input Twitter data file (JSON format) - required for file mode
- `--output`: Path to output directory for results
- `--spark-mode`: Spark execution mode ('local' or 'cluster')
- `--partitions`: Number of partitions for Spark (default: 4)
- `--skip-visualization`: Skip visualization generation

**Stream Mode:**
- `--mode stream`: Real-time streaming mode
- `--keyword`: Keyword/topic to stream tweets for - required for stream mode
- `--output`: Path to output directory for results
- `--batch-interval`: Streaming batch interval in seconds (default: 3)
- `--spark-mode`: Spark execution mode ('local' or 'cluster')
- `--partitions`: Number of partitions for Spark (default: 4)

## Data Format

Input data should be in JSON format (JSON Lines - one JSON object per line) with the following structure:

```json
{"id": "tweet_id", "text": "tweet content", "created_at": "2024-01-01T00:00:00Z", "user": {"screen_name": "username", "followers_count": 1000}, "retweet_count": 10, "favorite_count": 50}
{"id": "tweet_id2", "text": "another tweet", ...}
```

Or standard JSON array format:
```json
[
  {
    "id": "tweet_id",
    "text": "tweet content",
    "created_at": "2024-01-01T00:00:00Z",
    "user": {
      "screen_name": "username",
      "followers_count": 1000
    },
    "retweet_count": 10,
    "favorite_count": 50
  }
]
```

## Analysis Modules

### Sentiment Analysis
- Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Combines with TextBlob for enhanced accuracy
- Outputs: sentiment label (positive/negative/neutral) and confidence score

### Factuality Detection
- Keyword-based fact-checking indicators
- User credibility metrics (based on followers and engagement)
- Engagement pattern analysis
- Outputs: factuality score (0-1) and reliability label (high/medium/low)

## Results

The analysis generates:
- **Sentiment distribution charts** (pie chart and bar chart)
- **Factuality distribution charts**
- **Sentiment vs Factuality correlation plots**
- **Engagement analysis visualizations**
- **Word clouds** (overall and by sentiment)
- **HTML summary report** with comprehensive statistics
- **CSV file** with detailed analysis results

All results are saved in the specified output directory (default: `results/`).

## Performance

- Optimized for large datasets (millions of tweets)
- Uses Spark's distributed computing with pandas UDFs for efficient processing
- Supports both local and cluster execution
- Adaptive query execution enabled for optimal performance

## License

This project is for educational purposes as part of Principles of Big Data Management course.

## Author

PBDM Project - Multimodal Twitter Data Analysis