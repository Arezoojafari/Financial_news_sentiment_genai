# LLM-Powered Sentiment Analysis for Real-Time Financial News  
**Google GenAI Intensive Project | Gemini + FAISS + Few-Shot Prompting**

**Author:** *Arezoo Jafari*  

---

## Project Summary

This project addresses the challenge of real-time financial sentiment analysis by leveraging Google’s Gemini 2.0 Flash model in combination with semantic search and few-shot prompting. The solution automatically retrieves financial news, finds the most semantically similar labeled example using FAISS, and prompts the LLM to classify sentiment with grounded context, returning structured and interpretable results.

---

## Problem Statement

Financial analysts and portfolio managers often ask:

> “How did the market perform today?”  
> “What’s the sentiment around a specific company or sector?”

Answering these questions manually by reading dozens of news articles is time-consuming, inconsistent, and not scalable.

---

## Features

-  End-to-end GenAI pipeline for real-time news sentiment detection  
-  Fast semantic retrieval using FAISS for top-1 example matching  
-  Few-shot prompting with real-world labeled financial data  
-  LLM-powered sentiment classification using Gemini  
-  Structured and parseable JSON output

---

##  How It Works

1. **Fetch real-time financial news** via NewsAPI based on a stock or market query.
2. **Embed** real-time and labeled news headlines.
3. **Index the labeled dataset with FAISS** to enable fast vector similarity search.
4. For each real-time headline:
   - Use **FAISS** to retrieve the top-1 semantically closest labeled example.
   - Use a sample of labeled examples to construct a **few-shot prompt**.
   - Send the prompt to **Gemini 2.0 Flash** for sentiment prediction.
5. Parse and return the results as structured **JSON**.

---

## Tech Stack

-  Google Gemini 2.0 Flash (LLM)
-  NewsAPI
-  Text embeddings
-  FAISS (Facebook AI Similarity Search for vector retrieval)
-  JSON (for structured outputs)


