# Vimarsh – AI-Validated Product Consensus Engine

Vimarsh is an AI/ML-powered full-stack application that extracts real product recommendations from Reddit discussions, validates them using an LLM, and presents explainable, source-backed results.

## 🚀 Features
- Reddit data ingestion with PRAW
- NLP-based product extraction (spaCy + heuristics)
- TF-IDF inspired consensus scoring
- LLM (Mistral) validation to prevent hallucinations
- Authenticity scoring (recency, sentiment, diversity)
- FastAPI backend with server-rendered UI
- CSV & JSON export
- Serverless-ready deployment

## 🧠 System Architecture
User Query  
→ Reddit Crawl  
→ NLP Product Extraction  
→ Consensus Scoring  
→ LLM Validation  
→ Explainable Results

## 🛠 Tech Stack
- Backend: FastAPI, Python
- NLP: spaCy, regex, heuristics
- AI: Mistral LLM
- Data: Reddit (PRAW)
- Frontend: Jinja2 + TailwindCSS
