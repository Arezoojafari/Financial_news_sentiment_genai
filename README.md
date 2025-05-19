# LLM-Powered Sentiment Analysis for Real-Time Financial News  
**Gemini + RAG + Few-Shot Prompting | Google GenAI Intensive Project**

## 🔍 Overview

This project addresses a common challenge in financial analytics: how can decision-makers quickly assess market sentiment without manually reading dozens of news articles?

Using Google’s **Gemini 2.0 Flash** model, this solution leverages **retrieval-augmented generation (RAG)** and **few-shot prompting** to classify the sentiment of real-time financial news headlines. It integrates LLM reasoning with real-world labeled data to produce structured, trustworthy outputs.

---

## Problem Statement

Financial analysts and portfolio managers often ask:

> “How did the market perform today?”  
> “What’s the sentiment around a specific stock?”

Answering such questions typically requires scanning through a large volume of unstructured news articles. This is **time-consuming**, **error-prone**, and **not scalable**.

---

## Objective

Design an **end-to-end GenAI pipeline** that:
- Fetches **real-time financial news** via NewsAPI,
- Matches each headline with the **most semantically similar labeled example**,
- Prompts the Gemini model using **few-shot learning**, and
- Outputs **sentiment predictions** in a structured JSON format.

---

## Methodology

### 1. Data Sources
- **Labeled Dataset**: Financial news headlines labeled as `positive`, `neutral`, or `negative`.
- **Real-Time News**: Fetched via [NewsAPI](https://newsapi.org/) using finance-related queries.

### 2. Semantic Retrieval
- Encode both real-time and labeled news headlines using sentence embeddings.
- Use FAISS to perform fast approximate nearest neighbor search on normalized embedding vectors to retrieve the most semantically similar labeled example for each real-time headline.

### 3. Prompt Construction (Few-Shot)
- Randomly sample 10 labeled examples.
- Include the 10 examples plus the matched example in a dynamically constructed prompt.
- Send this prompt to Gemini 2.0 Flash for prediction.

### 4. Model Output
- Receive structured JSON with:
  - Real-time news headline
  - Most similar labeled example
  - Sentiment predicted by Gemini
  - Sentiment of the matched example

---

## Output Example

```json
[
  {
    "real_time_news": "Federal Reserve signals rate hike pause.",
    "matched_example": "Fed holds rates steady amid economic slowdown.",
    "predicted_sentiment": "neutral",
    "matched_sentiment": "neutral"
  }
]
```
---
### Tech Stack

-  Google Gemini 2.0 Flash (LLM)
-  NewsAPI
-  Sentence-Transformers
-  FAISS (Facebook AI Similarity Search for vector retrieval)
-  Python
-  JSON for structured outputs

