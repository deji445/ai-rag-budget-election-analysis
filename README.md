# 📊 RAG-Based AI Assistant for Ghana Election & Budget Analysis

##  Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) AI assistant** that answers questions based on:

* Ghana Election Results (CSV)
* 2025 Budget Statement (PDF)

The system retrieves relevant data and uses a Large Language Model (LLM) to generate **accurate, context-grounded responses**.

---

##  Objectives

* Build a **manual RAG pipeline** (no LangChain or pre-built frameworks)
* Improve retrieval accuracy using:

  * Hybrid search
  * Query expansion
  * Domain-specific scoring
* Reduce hallucination through structured prompting
* Provide explainable outputs with retrieved context

---

##  System Architecture

```
User Query
     ↓
Query Expansion
     ↓
Embedding Model
     ↓
Hybrid Retrieval (Vector + Keyword)
     ↓
Reranking (Domain-Specific Scoring)
     ↓
Top-K Relevant Chunks
     ↓
Prompt Builder
     ↓
LLM (OpenAI / Gemini)
     ↓
Final Answer (Streamlit UI)
```

---

##  Features

###  Hybrid Retrieval

* Combines:

  * FAISS vector similarity
  * Keyword matching
* Improves both semantic and exact-match retrieval

---

###  Query Expansion

* Expands queries using synonyms
* Example:

  * "winner" → "won", "victor"
* Improves performance on paraphrased questions

---

###  Domain-Specific Reranking ⭐

* Boosts relevance using:

  * Vote percentage
  * Region matching
  * Year matching
* Ensures correct identification of election winners

---

###  Post-Processing Reasoning (Innovation) ⭐

* Automatically identifies the **top candidate** based on votes
* Injects structured hint into prompt
* Improves answer consistency and accuracy

---

###  Multi-LLM Support

Supports:

* OpenAI (GPT models)
* Google Gemini

Switch using environment variables.

---

##  Project Structure

```
project/
│
├── data/
│   ├── Ghana_Election_Result.csv
│   └── 2025-Budget-Statement.pdf
│
├── src/
│   ├── data_loader.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── retrieval.py
│   ├── pipeline.py
│   ├── prompt.py
│   └── llm.py
│
├── app.py
└── README.md
```

---

##  How to Run

### 1. Install dependencies

```
pip install streamlit pandas faiss-cpu sentence-transformers PyPDF2 google-generativeai openai
```

---

### 2. Set API Key

#### For Gemini:

```
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="your_key_here"
```

#### For OpenAI:

```
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your_key_here"
```

---

### 3. Run the app

```
streamlit run app.py
```

---

##  Example Queries

###  Supported Queries

* Who won in Savannah Region in 2020?
* What percentage of votes did John Mahama receive in 2020?
* Who was victorious in Savannah Region in 2020?

---

###  Adversarial Queries

* Who won the election in Ketu North?
  → Dataset does not contain constituency data

* Who won the election?
  → Ambiguous query (missing year and region)

---

##  Evaluation Summary

| Query Type               | Result                   |
| ------------------------ | ------------------------ |
| Specific factual queries | ✅ Accurate               |
| Paraphrased queries      | ✅ Handled well           |
| Unsupported queries      | ✅ Returns "I don't know" |
| Ambiguous queries        | ✅ Avoids assumptions     |

---

##  Key Insights

* RAG improves factual accuracy compared to standalone LLMs
* Retrieval quality directly affects answer correctness
* Domain-aware scoring significantly improves performance
* Prompt constraints reduce hallucination

---

##  Limitations

* Dataset is limited to **regional-level data**
* Cannot answer constituency-level questions
* Some irrelevant chunks may still appear in lower rankings

---

##  Future Improvements

* Add constituency-level dataset
* Improve entity recognition (NER)
* Implement feedback-based learning
* Add conversational memory

---

## 👨 Author

Damilola Ayodeji
BSc Information Technology
Academic City University College

---

##  Notes

* This project was built **without using LangChain or LlamaIndex**
* All retrieval, chunking, and prompting logic were implemented manually

---
