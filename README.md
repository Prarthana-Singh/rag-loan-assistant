# 🤖 RAG Q&A Chatbot for Loan Dataset

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that answers questions intelligently using a structured loan dataset (`Training.csv`). 

### Key Features
- **FAISS** for semantic search to quickly retrieve relevant information from the dataset.
- **FLAN-T5** (Hugging Face) for natural language text generation.
- Handles **loan-related queries** with accurate and context-aware answers.
- Combines **retrieval-based reasoning** with **generative AI**, suitable for:
  - Financial data insights
  - Customer support automation
  - Educational purposes

---

### 🎥 Demo
![RAG Chatbot Demo](https://github.com/user-attachments/assets/5a0a3e02-f3bb-48c3-b8cc-f8d1a999f02c)

---

### 🖼 Demo Screenshots
<img width="1919" height="901" alt="Screenshot 1" src="https://github.com/user-attachments/assets/cca541a9-3eb0-477b-8393-35db56401802" />

<img width="1919" height="927" alt="Screenshot 2" src="https://github.com/user-attachments/assets/e5fb925a-2e9a-4f72-80e1-e94c90d943d5" />

<img width="1919" height="936" alt="Screenshot 3" src="https://github.com/user-attachments/assets/9ebd95d0-409f-40fc-b88a-9e00a487043e" />

---

### ⚡ Optional Sections to Add (to make it portfolio-ready)
1. **Installation / Setup**  
2. **Usage Instructions**  
3. **Tech Stack**  
4. **Future Improvements**  


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
