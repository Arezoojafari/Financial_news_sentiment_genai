import os
import time
import numpy as np
import pandas as pd
import faiss
import random
from tqdm import tqdm
from typing import List
from typing import Dict,Any
import requests
from google import genai
from database.database import load_index_and_ids
import sqlite3


MODEL_ID="gemini-2.0-flash-lite"

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# Load embeddings and build FAISS index at startup
faiss_index, index_to_dbid = load_index_and_ids("database/labeled_news.db")


def get_gemini_embedding(text):
    try:
        response = client.models.embed_content(
                model="models/text-embedding-004",
                contents=text,
                config={"task_type":'RETRIEVAL_DOCUMENT'}
        )
        return response.embeddings
    except Exception as e:
        print("Embedding failed:", e)
        return None
        

def fetch_news(query: str, max_results: int) -> List[str]:
    """
    Fetches recent English-language news articles matching the provided search query using the NewsAPI.

    Args:
        query (str): The search term or keywords to look for in news articles.
        max_results (int, optional): The maximum number of articles to retrieve. Defaults to 5.

    Returns:
        List[str]: A list of strings, where each string contains the title and description of a news article.
                   Returns an empty list if the request fails or no articles are found.

    Example:
        articles = fetch_news("artificial intelligence", max_results=10)
        # Returns a list of up to 10 news articles about artificial intelligence.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": max_results,
        "sortBy": "relevance",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Failed to fetch news:", response.text)
        return []
    return [f"{a['title']}. {a['description']}" for a in response.json().get("articles", []) if a['description']]


def retrieve_similar_labeled_example(news_text: str) -> Dict[str, str]:
    """
    Finds and returns the most similar labeled financial example to the given news article using FAISS-based semantic search.

    Args:
        news_text (str): The news article or headline text to search for similar labeled examples.

    Returns:
        Dict[str, str]: A dictionary containing the following keys:
            - 'example_text': The text of the most similar labeled finance example from the database.
            - 'example_label': The label or category associated with the matched example.
            - If embedding fails, returns {'error': 'Failed to embed'}.

    Example:
        result = retrieve_similar_labeled_example("Apple stock surges after earnings report.")
        # Returns: {'example_text': 'Apple posts record quarterly revenue...', 'example_label': 'positive'}
    """
    query_vec = get_gemini_embedding(news_text)
    if query_vec is None:
        return {"error": "Failed to embed"}
    query_array = np.array(query_vec[0].values, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(query_array)
    D, I = faiss_index.search(query_array, 1)
    db_id = index_to_dbid[I[0][0]]
    # Fetch from SQLite
    conn = sqlite3.connect("database/labeled_news.db")
    c = conn.cursor()
    c.execute("SELECT text, label FROM news WHERE id = ?", (db_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"example_text": row[0], "example_label": row[1]}
    else:
        return {"error": "Not found in DB"}


def classify_sentiment_with_few_shot(news: str, example_text: str, example_label: str) -> Dict[str, str]:
    """
    Classifies the sentiment of a financial news article (positive, negative, or neutral) using a few-shot prompt with Gemini.
    The function leverages a matched labeled example as in-context reference to improve classification accuracy.

    Args:
        news (str): The text of the financial news article to be classified.
        example_text (str): A labeled example news text that is semantically similar to the input.
        example_label (str): The sentiment label ('positive', 'negative', or 'neutral') of the example_text.

    Returns:
        Dict[str, str]: A dictionary containing:
            - 'sentiment': The predicted sentiment for the input news article ('positive', 'negative', 'neutral', or 'unknown').
            - 'news': The input news article text.
            - 'example_text': The matched example news text used for few-shot prompting.
            - 'example_label': The sentiment label of the matched example.
            - If an error occurs, returns {'error': <error_message>}.

    Example:
        result = classify_sentiment_with_few_shot(
            news="Tesla shares drop after recall announcement.",
            example_text="Tesla faces scrutiny after software glitch, stock falls.",
            example_label="negative"
        )
        # Returns: {
        #   'sentiment': 'negative',
        #   'news': "...",
        #   'example_text': "...",
        #   'example_label': "negative"
        # }
    """
    prompt = f"""You are a financial sentiment classifier.
    Here is an example:
    Text: {example_text}
    Sentiment: {example_label.capitalize()}
    
    Now classify the following:
    Text: {news}
    Sentiment:
    """

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        sentiment = response.text.strip().lower()

        if sentiment.startswith("positive"):
            sentiment = "positive"
        elif sentiment.startswith("negative"):
            sentiment = "negative"
        elif sentiment.startswith("neutral"):
            sentiment = "neutral"
        else:
            sentiment = "unknown"

        return {
            "sentiment": sentiment,
            "news": news,
            # "example_text": example_text,
            # "example_label": example_label
        }

    except Exception as e:
        return {"error": str(e)}
