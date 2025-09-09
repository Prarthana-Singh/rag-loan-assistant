# 🤖 RAG Q&A Chatbot for Loan Dataset

This project implements a Retrieval-Augmented Generation (RAG) chatbot that intelligently answers questions based on a structured loan dataset (`Training.csv`). It combines semantic search using FAISS with a lightweight text generation model from Hugging Face (`FLAN-T5`).

---

## 🔧 Installation

1. Clone the repo or download the project folder.
2. Place your `Training.csv` dataset in a folder named `data/`.
3. Install dependencies:
```bash
pip install -r requirements.txt
````

---

## 🚀 Running the Chatbot

```bash
python main.py
```

You’ll see:

```
RAG Chatbot ready. Type your loan-related questions!
You:
```

Start asking questions like:

* `"What does credit history mean?"`
* `"Explain a loan from an urban area."`
* `"Compare loans with high applicant income."`

---

## 🧠 How It Works

### 1. Document Creation

Each row from `Training.csv` is converted into a Q\&A-style document string, which is used for retrieval.

### 2. Embedding + Indexing

Documents are embedded using `sentence-transformers/all-MiniLM-L6-v2`, and stored in a FAISS index for similarity search.

### 3. Retrieval + Generation

On user input:

* Top-`k` matching documents are retrieved.
* A prompt is crafted combining the context and user query.
* `google/flan-t5-small` or `flan-t5-base` generates the final response.

---

## ❓ FAQ Support (Optional)

Basic definitions for terms like "Credit History" or "Loan Status" are handled by a simple keyword-based fallback if no good document matches.

---

## 📦 Example Output

**User:** What does Credit History mean?
**Bot:** Credit history indicates whether a borrower has repaid past credit obligations. A value of 1.0 means good credit history; 0.0 means bad or no credit history.

---

## 📌 Dependencies

```
transformers
sentence-transformers
faiss-cpu
torch
pandas
```

---

## ✨ Future Improvements

* Add Gradio web interface.
* Use larger generative models (e.g., Mistral-7B) for better answers.
* Support multilingual queries.
* Add more domain knowledge in dataset.

---

## 🧑‍💻 Author

Built by [Prarthana Singh](https://www.linkedin.com/in/prarthanasingh/)
Project for Celebal Internship Week 8 – RAG Systems
