#  NLP Project 2024/2025

This repository demonstrates a **complete natural language processing (NLP) pipeline**, built around the [RAG-12000 dataset](https://huggingface.co/datasets/neural-bridge/rag-dataset-12000).  
It progresses from traditional text preprocessing and clustering to **large language model (LLM)** inference and a **voice-enabled retrieval-augmented generation (RAG)** interface.

>  **Goal:** Showcase practical expertise in modern NLP — from text cleaning to semantic search and multimodal AI — using real data and state-of-the-art models.

---

##  Highlights

- **End-to-End NLP Pipeline:** Preprocessing → embeddings → clustering → classification → retrieval → generation → speech interface  
- **Bridging Classical & Modern NLP:** TF-IDF, GloVe, and K-Means alongside T5, Sentence-BERT, and LLaMA-2  
- **Applied LLMs:** Retrieval-Augmented Generation with T5 and fine-tuned LLaMA-2  
- **Interactive AI System:** Voice-based RAG chatbot combining Whisper (STT), Gemini via LangChain (retrieval), and Tacotron2 (TTS)  
- **Comprehensive Evaluation:** BLEU, F1, Exact Match, and Recall@K metrics to measure performance at every stage  
## The Pipeline Step-by-Step

Each notebook corresponds to a distinct stage in the workflow. Together, they build an **end-to-end AI system** that cleans, analyzes, understands, retrieves, and generates text.

| Notebook | Title | Description |
|-----------|--------|-------------|
| **01_preprocessing.ipynb** | *Data Preparation & Exploration* | Loads and inspects the RAG-12000 dataset. Performs cleaning (removing nulls, normalizing text, handling punctuation), computes corpus statistics, and visualizes distributions of question lengths and token counts. Establishes data consistency for downstream tasks. |
| **02_similarity_and_embeddings.ipynb** | *Lexical vs. Semantic Similarity* | Computes question–context similarity using both TF-IDF and GloVe embeddings. Highlights the difference between lexical overlap and semantic alignment, explaining why averaged embeddings sometimes misrepresent meaning. Introduces cosine similarity and visualization of similarity distributions. |
| **03_clustering.ipynb** | *Unsupervised Topic Discovery* | Uses TF-IDF vectors and K-Means clustering to uncover latent structure in the dataset. Includes silhouette analysis for optimal K, PCA/SVD visualizations, and top-terms per cluster. Demonstrates the limits of pure lexical clustering versus semantically aware embeddings. |
| **04_data_cleaning.ipynb** | *Refining the Dataset* | Removes duplicates, meta-questions, and noisy examples. Creates consistent training, validation, and test splits (70/15/15). Around 8% of low-quality or meta entries are discarded to improve model reliability. |
| **05_clustering_and_classification.ipynb** | *From Clusters to Supervised Labels* | Converts unsupervised clusters into pseudo-labels for supervised learning. Trains multiple text classifiers (Logistic Regression, SVM, and DistilBERT). Includes grid search for hyperparameter tuning and compares classical vs. transformer-based models. |
| **06_semantic_search.ipynb** | *Dense Vector Retrieval* | Builds a semantic search engine using Sentence-BERT and Cross-Encoder models. Compares retrieval quality using cosine similarity, HNSWLIB for approximate nearest neighbor search, and cross-encoder reranking. Stores and reuses embeddings for scalable search. |
| **07_t5.ipynb** | *Text-to-Text Generation with T5* | Fine-tunes and evaluates a T5-small model for question answering. Compares zero-shot, few-shot, and fine-tuned scenarios. Uses Exact Match, F1, and BLEU metrics. Shows that providing relevant context significantly improves fluency and factual accuracy. |
| **08_t5_rag.ipynb** | *Retrieval-Augmented Generation (RAG)* | Integrates Sentence-BERT retrieval with T5 generation. Evaluates both retrieval recall and answer quality. Experiments with recall@k, answer accuracy, and latency trade-offs. Demonstrates the synergy between retrieval grounding and generative fluency. |
| **09_LLM.ipynb** | *LLMs for Question Answering (LLaMA-2)* | Explores **LLaMA-2-7B-Chat (4-bit)** in zero-shot, one-shot, and fine-tuned setups. Defines generation and evaluation utilities (BLEU, EM, F1, semantic similarity). Compares LLaMA’s output quality against T5 and highlights benefits of model quantization for local inference. |
| **10_LLM_inference.ipynb** | *Optimized LLM Inference* | Loads the fine-tuned weights of LLaMA-2 and tests real examples. Demonstrates how fine-tuning improves consistency and reduces hallucinations. Focuses on inference-time optimization and output post-processing. |
| **11_s2t_news_rag_t2s_gui.ipynb** | *Voice-Enabled News RAG Interface* | Combines Speech-to-Text (Whisper), LangChain RAG agent (Gemini API), and Text-to-Speech (Tacotron2 + WaveGlow) into an interactive Gradio app. Users can speak queries, retrieve recent news, and hear generated summaries aloud. It’s a functional AI assistant prototype connecting multiple modalities. |
