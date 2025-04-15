# Financial-Sentiment-Analyser
A real-time financial sentiment analysis app that uses FinBERT to classify live Yahoo Finance news and user input into positive, negative, or neutral sentiment with insightful visualizations.


🔧 Features

🧠 FinBERT Sentiment Model: A transformer-based model specifically trained on financial data for accurate sentiment classification.

🔍 Live Web Scraping: Fetches up-to-date articles from Yahoo Finance using Selenium and BeautifulSoup.

📊 Visual Analytics: Displays sentiment distribution with interactive charts using Matplotlib and Seaborn.

🧾 User Input Analysis: Users can enter custom financial statements to analyze sentiment instantly.

⏱️ Timeout Handling: Built-in timeout to ensure scraping does not run indefinitely.

🧹 Noise Filtering: Filters out irrelevant or error-prone content using custom heuristics and keyword matching.

💾 CSV Export (Optional): Can store fetched article content for offline processing or evaluation.

🛠️ Setup & Installation:
1. Clone the repository
2. Install requirements: pip install -r requirements.txt
3. Run the Streamlit app: streamlit run app.py
