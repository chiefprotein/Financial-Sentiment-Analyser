import re
import time
import csv
import signal
import os
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Timeout handler to stop script after 3 minutes
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# Set a 3-minute timeout (180 seconds)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(180)  # 3-minute timer


# ChromeDriver path (Ensure you have the correct path to chromedriver)
CHROMEDRIVER_PATH = "/Users/jenil/Downloads/chromedriver-mac-arm64/chromedriver"  # Update with your actual path

# Initialize headless Chrome
options = Options()
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Start ChromeDriver service
service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=options)

# Define financial terms to filter relevant articles
financial_terms = [
    "stock", "market", "investment", "finance", "earnings", "IPO",
    "cryptocurrency", "bitcoin", "NASDAQ", "NYSE", "share", "trading",
    "revenue", "profit", "loss", "bond", "inflation", "interest rate",
    "dividend", "capital", "funds", "portfolio", "forecast", "analyst"
]

# Phrases to ignore in the final CSV
ignore_phrases = [
    "Tip: Try a valid symbol or a specific company name for relevant results",
    "Sign in to access your portfolio",
    "Oops, something went wrong",
    "Try again."
]

# URL to scrape Yahoo Finance
base_url = "https://finance.yahoo.com/"

# Start Selenium browser
driver.get(base_url)
time.sleep(5)

# Parse main page content
soup = BeautifulSoup(driver.page_source, 'html.parser')

# Find all article links
articles = soup.find_all('a', href=True)

# Define regex pattern for valid finance article links
finance_url_pattern = re.compile(r"https://finance\.yahoo\.com/news/")

# Store article data
news_data = []

try:
    # Loop through all articles
    for article in articles:
        link = article['href']
        title = article.text.strip()

        # Handle relative or absolute URLs
        full_url = urljoin(base_url, link)

        # Validate if it's a Yahoo Finance article
        if finance_url_pattern.match(full_url):
            print(f"Attempting to load article: {full_url}")

            # Open article using Selenium
            driver.get(full_url)
            time.sleep(5)

            # Parse article content
            article_soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Extract paragraphs from the article
            paragraphs = article_soup.find_all('p')
            sentences = [p.text.strip() for p in paragraphs if p.text.strip()]

            # Validate financial content
            content_preview = " ".join(sentences[:3]).lower()
            if (
                any(term in title.lower() for term in financial_terms) or
                any(term in content_preview for term in financial_terms)
            ):
                # Check minimum length
                if len(sentences) >= 5 and len(" ".join(sentences).split()) >= 150:
                    news_data.extend(sentences)
                    print(f"Article added: {full_url}")
                else:
                    print(f"Skipped short or non-financial article: {full_url}")
            else:
                print(f"Skipped article with no financial terms: {full_url}")
        else:
            print(f"Skipped non-finance article: {full_url}")

except TimeoutException:
    print("Time limit reached. Stopping article collection...")

except Exception as e:
    print(f"Error occurred: {e}")

# Disable the timer after data collection
signal.alarm(0)

# Filter out unwanted phrases before writing to CSV
filtered_data = [
    sentence for sentence in news_data
    if not any(ignore_phrase in sentence for ignore_phrase in ignore_phrases)
]
print(filtered_data)

# Define local output file path
output_file = os.path.join(os.getcwd(), "finance_articles.csv")

# Save filtered articles to a CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sentence'])
    for sentence in filtered_data:
        writer.writerow([sentence])

print(f"✅ Extraction complete! Data saved in '{output_file}'.")

# Close browser
driver.quit()
