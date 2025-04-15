import streamlit as st
import pandas as pd
import subprocess
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Set page title
st.set_page_config(page_title="Financial Sentiment Analyzer", layout="wide")

# 🔹 Load FinBERT Model
@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained('./FinBERT')
    model = AutoModelForSequenceClassification.from_pretrained('./FinBERT')
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation=True, max_length=1024)

sentiment_pipeline = load_pipeline()

# 🔹 Trigger realltimedata.py and wait for it to finish
def run_realltimedata():
    """Runs realltimedata.py to scrape Yahoo Finance and save data to finance_articles.csv."""
    st.info("🔄 Running real-time data scraper...")
    try:
        # Run realltimedata.py as a subprocess
        subprocess.run(["python3", "realltimedata.py"], check=True)
        st.success("✅ Data scraping completed successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Error while running realltimedata.py: {e}")
        return False
    return True

# 🔹 Read finance_articles.csv and perform sentiment analysis
def analyze_finance_articles():
    """Reads finance_articles.csv and performs sentiment analysis."""
    csv_path = "finance_articles.csv"
    if not os.path.exists(csv_path):
        st.error("❌ finance_articles.csv not found. Please run the scraper first.")
        return None

    # Load the CSV file
    st.info("📂 Loading scraped data from finance_articles.csv...")
    df = pd.read_csv(csv_path)

    if df.empty:
        st.warning("⚠️ The CSV file is empty. No data to analyze.")
        return None

    # Perform sentiment analysis
    st.info("🔍 Performing sentiment analysis on the scraped data...")
    predictions = sentiment_pipeline(df['Sentence'].tolist())
    df['Sentiment'] = [pred['label'] for pred in predictions]

    return df

# 🔹 Main Streamlit App
st.title("📊 Financial Sentiment Analyzer")
st.write("Analyze the sentiment of financial news articles in real-time using FinBERT.")

# Sidebar: Trigger scraper and analyze data
if st.sidebar.button("Run Scraper and Analyze"):
    # if run_realltimedata():
        news_df = analyze_finance_articles()
        if news_df is not None:
            # Display sentiment breakdown
            st.header("📊 Sentiment Breakdown of Latest Financial News")
            fig, ax = plt.subplots(figsize=(6, 4))
            sentiment_counts = news_df['Sentiment'].value_counts()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["red", "green", "blue"], ax=ax)
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

            # Display articles with sentiment
            st.subheader("🔍 Latest Financial News Sentiment Analysis")
            st.dataframe(news_df[['Sentence', 'Sentiment']], use_container_width=True)
        else:
            st.warning("⚠️ No data available for analysis.")
else:
    st.info("Click the button in the sidebar to run the scraper and analyze data.")

# 🔹 Sidebar: User Input Sentiment Analysis
st.sidebar.header("User Input Sentiment Analysis")
user_input = st.sidebar.text_area("Enter a financial news headline or text:")

if st.sidebar.button("Analyze Sentiment"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)[0]
        st.sidebar.success(f"Predicted Sentiment: **{result['label'].capitalize()}** ({result['score']:.2f} confidence)")
    else:
        st.sidebar.warning("Please enter some text.")