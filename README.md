# Transformer-Based News Summarization Using BART

This project implements an automatic news summarization system using the pre-trained BART Transformer model from Hugging Face. The system generates concise summaries from long-form news articles while preserving the main ideas and context.

---

# Project Overview

* **Task:** Abstractive Text Summarization
* **Model:** BART (`facebook/bart-large-cnn`)
* **Libraries:** PyTorch, Hugging Face Transformers, Gradio
* **Language:** Python

The goal of this project is to demonstrate how a pre-trained Transformer model can be used to automatically summarize lengthy news articles into shorter and more readable content.

---

# Interpretability

Although the primary focus of this project is text summarization, the generated summaries can be analyzed by comparing:

* Original article length
* Summary length
* Compression ratio
* Preservation of key information
* Readability of generated summaries

This provides insight into how Transformer-based models condense information while maintaining important context.

---

# Installation

Install required packages:

```
pip install transformers torch gradio
```

# Usage

Run the summarizer:

```
python summarizer.py
```

After launching, open the local Gradio URL(http://127.0.0.1:7860) in your browser.

Steps:

1. User pastes a news article into the Gradio interface.
2. The article is processed by the pre-trained BART model.
3. BART generates an abstractive summary. 
4. The system calculates:
Original Word Count, Summary Word Count, Compression Ratio
5. Results are displayed through the web interface. 

---

# How It Works

```text
News Article
      ↓
Gradio Web Interface
      ↓
BART Transformer Model
      ↓
Generated Summary
      ↓
Word Count & Compression Ratio Analysis
```

The system uses the pre-trained `facebook/bart-large-cnn` model from Hugging Face to perform abstractive text summarization. The model reads the input article, identifies important information, and generates a concise summary in natural language.

---

# Demo Video

YouTube Demo: https://youtu.be/5gE61xNXGu4?si=Ect6TCKxhlUbp9o8
