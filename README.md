LLM-Powered Sentiment Analysis for Real-Time Financial News
Gemini + RAG + Few-Shot Prompting | Google GenAI Intensive Project

🔍 Overview
This project addresses a common challenge in financial analytics: how can decision-makers quickly assess market sentiment without manually reading dozens of news articles?

Using Google’s Gemini 2.0 Flash model, this solution leverages retrieval-augmented generation (RAG) and few-shot prompting to classify the sentiment of real-time financial news headlines. It integrates LLM reasoning with real-world labeled data to produce structured, trustworthy outputs.

Problem Statement
Financial analysts and portfolio managers often ask questions like:

“How did the market perform today?” or “What’s the sentiment around a specific stock?”

However, answering such questions typically requires scanning through a large volume of unstructured news articles. This is time-consuming, error-prone, and not scalable.

Objective
Design an end-to-end GenAI pipeline that:

Fetches real-time financial news via NewsAPI,

Matches each headline with the most semantically similar labeled example,

Prompts a Gemini model using few-shot learning, and

Outputs sentiment predictions in a structured JSON format.

Methodology
1. Data Sources
Labeled Dataset: A financial news dataset containing text and sentiment labels (positive, neutral, negative).

Real-Time News: Collected dynamically via NewsAPI based on market-related queries.

2. Semantic Retrieval
Each real-time headline is embedded using vector embeddings.

Cosine similarity is used to find the most semantically similar example from the labeled dataset.

3. Few-Shot Prompting
A sample of 10 labeled news items is dynamically inserted into the prompt as examples.

The real-time headline and its top match are then fed to the Gemini model with clear task instructions.

4. Gemini Model Response
The model returns:

Predicted sentiment for the real-time headline

Sentiment of the matched example

Structured output in JSON format



Technologies & Tools
Gemini 2.0 Flash API (Google) – for LLM inference

NewsAPI – for real-time news scraping

Sentence Transformers / Embeddings – for semantic similarity

Cosine Similarity – for top-1 retrieval

Python – primary programming language

JSON – for structured output

Key Features
End-to-end pipeline: from data retrieval to structured sentiment output

RAG-enhanced: grounding Gemini’s outputs with real-world labeled data

Few-shot learning: dynamically constructed prompts tailored per query

Scalable architecture: designed for extension to multi-news workflows

Takeaways
LLMs alone are not enough: Grounding via RAG and example-based prompting improves accuracy, especially in domain-specific contexts like finance.

Embedding space matters: Matching real-time news to semantically similar labeled texts helps guide the model’s behavior toward realistic and consistent outputs.

Fine prompt engineering is critical: Prompt formatting, example clarity, and instruction quality directly impact the precision of sentiment classification.
