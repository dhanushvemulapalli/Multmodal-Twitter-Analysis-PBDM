"""
Visualization Module
Creates visualizations and reports from analysis results
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, when
import numpy as np


class Visualizer:
    """Creates visualizations and reports"""
    
    def __init__(self, spark: SparkSession, output_dir: str = "results"):
        """
        Initialize visualizer
        
        Args:
            spark: SparkSession instance
            output_dir: Directory to save visualizations
        """
        self.spark = spark
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _spark_to_pandas(self, df, limit: int = 10000):
        """
        Convert Spark DataFrame to Pandas (for visualization)
        
        Args:
            df: Spark DataFrame
            limit: Maximum number of rows to collect
            
        Returns:
            Pandas DataFrame
        """
        count = df.count()
        if count > limit:
            print(f"Warning: Dataset has {count} rows. Sampling {limit} rows for visualization.")
            df = df.sample(False, limit / count)
        
        return df.toPandas()
    
    def plot_sentiment_distribution(self, df, save_path: str = None):
        """
        Plot sentiment distribution
        
        Args:
            df: Spark DataFrame with sentiment analysis results
            save_path: Path to save the plot
        """
        # Aggregate sentiment counts
        sentiment_counts = df.groupBy("sentiment_label").agg(
            count("*").alias("count")
        ).orderBy("sentiment_label")
        
        df_pandas = sentiment_counts.toPandas()
        
        # Create pie chart
        fig = px.pie(
            df_pandas,
            values='count',
            names='sentiment_label',
            title='Sentiment Distribution',
            color_discrete_map={
                'positive': '#2ecc71',
                'negative': '#e74c3c',
                'neutral': '#95a5a6'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        if save_path is None:
            save_path = self.output_dir / "sentiment_distribution.html"
        else:
            save_path = Path(save_path)
        
        fig.write_html(str(save_path))
        print(f"Sentiment distribution chart saved to {save_path}")
        
        # Also create a bar chart
        fig2 = px.bar(
            df_pandas,
            x='sentiment_label',
            y='count',
            title='Sentiment Distribution (Bar Chart)',
            color='sentiment_label',
            color_discrete_map={
                'positive': '#2ecc71',
                'negative': '#e74c3c',
                'neutral': '#95a5a6'
            }
        )
        
        bar_path = self.output_dir / "sentiment_distribution_bar.html"
        fig2.write_html(str(bar_path))
    
    def plot_factuality_distribution(self, df, save_path: str = None):
        """
        Plot factuality/reliability distribution
        
        Args:
            df: Spark DataFrame with factuality analysis results
            save_path: Path to save the plot
        """
        # Aggregate reliability counts
        reliability_counts = df.groupBy("reliability_label").agg(
            count("*").alias("count")
        ).orderBy("reliability_label")
        
        df_pandas = reliability_counts.toPandas()
        
        # Create visualization
        fig = px.bar(
            df_pandas,
            x='reliability_label',
            y='count',
            title='Factuality/Reliability Distribution',
            color='reliability_label',
            color_discrete_map={
                'high': '#27ae60',
                'medium': '#f39c12',
                'low': '#e74c3c'
            },
            category_orders={"reliability_label": ["high", "medium", "low"]}
        )
        
        if save_path is None:
            save_path = self.output_dir / "factuality_distribution.html"
        else:
            save_path = Path(save_path)
        
        fig.write_html(str(save_path))
        print(f"Factuality distribution chart saved to {save_path}")
    
    def plot_sentiment_vs_factuality(self, df, save_path: str = None):
        """
        Plot correlation between sentiment and factuality
        
        Args:
            df: Spark DataFrame with both sentiment and factuality results
            save_path: Path to save the plot
        """
        # Sample data for visualization
        df_sample = df.sample(False, 0.1).limit(5000)
        df_pandas = self._spark_to_pandas(df_sample)
        
        # Create scatter plot
        fig = px.scatter(
            df_pandas,
            x='sentiment_compound',
            y='factuality_score',
            color='sentiment_label',
            size='total_engagement',
            hover_data=['tweet_text'],
            title='Sentiment vs Factuality Analysis',
            labels={
                'sentiment_compound': 'Sentiment Score',
                'factuality_score': 'Factuality Score'
            }
        )
        
        if save_path is None:
            save_path = self.output_dir / "sentiment_vs_factuality.html"
        else:
            save_path = Path(save_path)
        
        fig.write_html(str(save_path))
        print(f"Sentiment vs Factuality chart saved to {save_path}")
    
    def plot_engagement_analysis(self, df, save_path: str = None):
        """
        Plot engagement metrics analysis
        
        Args:
            df: Spark DataFrame with engagement data
            save_path: Path to save the plot
        """
        # Calculate average engagement by sentiment
        engagement_by_sentiment = df.groupBy("sentiment_label").agg(
            avg("total_engagement").alias("avg_engagement"),
            avg("retweet_count").alias("avg_retweets"),
            avg("favorite_count").alias("avg_favorites")
        )
        
        df_pandas = engagement_by_sentiment.toPandas()
        
        # Create grouped bar chart
        fig = go.Figure()
        
        x = df_pandas['sentiment_label']
        fig.add_trace(go.Bar(
            x=x,
            y=df_pandas['avg_retweets'],
            name='Retweets',
            marker_color='#3498db'
        ))
        fig.add_trace(go.Bar(
            x=x,
            y=df_pandas['avg_favorites'],
            name='Favorites',
            marker_color='#9b59b6'
        ))
        fig.add_trace(go.Bar(
            x=x,
            y=df_pandas['avg_engagement'],
            name='Total Engagement',
            marker_color='#e67e22'
        ))
        
        fig.update_layout(
            title='Average Engagement by Sentiment',
            xaxis_title='Sentiment',
            yaxis_title='Average Count',
            barmode='group'
        )
        
        if save_path is None:
            save_path = self.output_dir / "engagement_analysis.html"
        else:
            save_path = Path(save_path)
        
        fig.write_html(str(save_path))
        print(f"Engagement analysis chart saved to {save_path}")
    
    def create_wordcloud(self, df, save_path: str = None, sentiment: str = None):
        """
        Create word cloud from tweets
        
        Args:
            df: Spark DataFrame with tweet text
            sentiment: Filter by sentiment (optional)
            save_path: Path to save the word cloud
        """
        # Filter by sentiment if specified
        if sentiment:
            df = df.filter(col("sentiment_label") == sentiment)
        
        print(df.printSchema())
        # Collect text
        texts = df.select("tweet_text").toPandas()["tweet_text"].dropna().tolist()
        combined_text = " ".join(texts)
        
        if not combined_text.strip():
            print("No text available for word cloud")
            return
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            collocations=False
        ).generate(combined_text)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        title = f'Word Cloud'
        if sentiment:
            title += f' - {sentiment.capitalize()} Sentiment'
        plt.title(title, fontsize=16)
        
        if save_path is None:
            sentiment_suffix = f"_{sentiment}" if sentiment else ""
            save_path = self.output_dir / f"wordcloud{sentiment_suffix}.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Word cloud saved to {save_path}")
    
    def generate_summary_report(self, df, save_path: str = None):
        """
        Generate comprehensive summary report
        
        Args:
            df: Spark DataFrame with all analysis results
            save_path: Path to save the report
        """
        if save_path is None:
            save_path = self.output_dir / "summary_report.html"
        else:
            save_path = Path(save_path)
        
        # Calculate statistics
        total_tweets = df.count()
        
        sentiment_stats = df.groupBy("sentiment_label").agg(
            count("*").alias("count"),
            avg("sentiment_confidence").alias("avg_confidence")
        ).toPandas()
        
        factuality_stats = df.groupBy("reliability_label").agg(
            count("*").alias("count"),
            avg("factuality_score").alias("avg_score")
        ).toPandas()
        
        overall_stats = df.agg(
            avg("factuality_score").alias("avg_factuality"),
            avg("sentiment_compound").alias("avg_sentiment"),
            avg("total_engagement").alias("avg_engagement")
        ).toPandas()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Twitter Analysis Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .stat-box {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Twitter Multimodal Analysis Summary Report</h1>
            
            <div class="stat-box">
                <h2>Overall Statistics</h2>
                <p><strong>Total Tweets Analyzed:</strong> {total_tweets:,}</p>
                <p><strong>Average Factuality Score:</strong> {overall_stats['avg_factuality'].iloc[0]:.3f}</p>
                <p><strong>Average Sentiment Score:</strong> {overall_stats['avg_sentiment'].iloc[0]:.3f}</p>
                <p><strong>Average Engagement:</strong> {overall_stats['avg_engagement'].iloc[0]:.2f}</p>
            </div>
            
            <h2>Sentiment Distribution</h2>
            <table>
                <tr>
                    <th>Sentiment</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Avg Confidence</th>
                </tr>
        """
        
        for _, row in sentiment_stats.iterrows():
            percentage = (row['count'] / total_tweets) * 100
            html_content += f"""
                <tr>
                    <td>{row['sentiment_label'].capitalize()}</td>
                    <td>{row['count']:,}</td>
                    <td>{percentage:.2f}%</td>
                    <td>{row['avg_confidence']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Factuality/Reliability Distribution</h2>
            <table>
                <tr>
                    <th>Reliability Level</th>
                    <th>Count</th>
                    <th>Percentage</th>
                    <th>Avg Score</th>
                </tr>
        """
        
        for _, row in factuality_stats.iterrows():
            percentage = (row['count'] / total_tweets) * 100
            html_content += f"""
                <tr>
                    <td>{row['reliability_label'].capitalize()}</td>
                    <td>{row['count']:,}</td>
                    <td>{percentage:.2f}%</td>
                    <td>{row['avg_score']:.3f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <p><em>Report generated using PySpark Multimodal Twitter Analysis</em></p>
        </body>
        </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Summary report saved to {save_path}")
    
    def save_results_csv(self, df, save_path: str = None):
        """
        Save analysis results to CSV
        
        Args:
            df: Spark DataFrame with all results
            save_path: Path to save CSV
        """
        print("Writing results to:", save_path)
        if save_path is None:
            save_path = self.output_dir / "analysis_results.csv"
        else:
            save_path = Path(save_path)
        
        # Sample if too large
        count = df.count()
        if count > 100000:
            print(f"Dataset has {count} rows. Sampling 100,000 rows for CSV export.")
            df = df.sample(False, 100000 / count).limit(100000)

        
        df.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(save_path.parent / "csv_temp"))
        
        # Move the actual CSV file
        import glob
        import shutil
        csv_files = glob.glob(str(save_path.parent / "csv_temp" / "part-*.csv"))
        if csv_files:
            shutil.move(csv_files[0], str(save_path))
            shutil.rmtree(str(save_path.parent / "csv_temp"), ignore_errors=True)
        
        print(f"Results CSV saved to {save_path}")
