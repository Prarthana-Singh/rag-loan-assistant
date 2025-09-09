# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import faiss
# import pickle
# import os

# def embed_documents(documents):
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(documents, convert_to_tensor=False)
#     return embeddings

# def load_loan_documents(csv_path):
#     df = pd.read_csv(csv_path)
#     documents = []
#     for _, row in df.iterrows():
#         doc = (
#             f"Loan ID: {row['Loan_ID']}, Gender: {row['Gender']}, Married: {row['Married']}, "
#             f"Dependents: {row['Dependents']}, Education: {row['Education']}, "
#             f"Self Employed: {row['Self_Employed']}, Applicant Income: {row['ApplicantIncome']}, "
#             f"Coapplicant Income: {row['CoapplicantIncome']}, Loan Amount: {row['LoanAmount']}, "
#             f"Loan Term: {row['Loan_Amount_Term']}, Credit History: {row['Credit_History']}, "
#             f"Property Area: {row['Property_Area']}, Loan Status: {row['Loan_Status']}"
#         )
#         documents.append(doc)
#     return documents

# def build_faiss_index(documents, index_path="embeddings/faiss_index.bin"):
#     os.makedirs("embeddings", exist_ok=True)
#     embeddings = embed_documents(documents)
#     dim = len(embeddings[0])
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     faiss.write_index(index, index_path)
#     with open("embeddings/docs.pkl", "wb") as f:
#         pickle.dump(documents, f)

# def search_index(query, top_k=3):
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     query_embedding = model.encode([query])
#     index = faiss.read_index("embeddings/faiss_index.bin")
#     with open("embeddings/docs.pkl", "rb") as f:
#         documents = pickle.load(f)
#     _, indices = index.search(query_embedding, top_k)
#     return [documents[i] for i in indices[0]]












# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# # Load FLAN-T5 pipeline
# qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load and preprocess CSV into chunks
# def load_chunks(csv_path):
#     df = pd.read_csv(csv_path).fillna("NA")
#     rows = ["\n".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]
#     custom_chunks = [
#         "Loan_Status is the target variable. It is 'Y' for approved and 'N' for rejected loans.",
#         "Married applicants are slightly more likely to get loans, possibly due to combined incomes or coapplicants.",
#         "Unmarried applicants with good income and credit history can still get loans approved.",
#         "Credit_History is one of the strongest indicators of loan approval in the dataset.",
#         "ApplicantIncome and CoapplicantIncome together affect loan eligibility.",
#         "Education and Property_Area may also impact loan decisions slightly.",
#     ]
#     return custom_chunks + rows

# # Build FAISS index
# def build_index(chunks):
#     embeddings = model.encode(chunks)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings))
#     return index, chunks

# # Retrieve top relevant chunks
# def retrieve(query, index, chunks, k=3):
#     q_embed = model.encode([query])
#     D, I = index.search(np.array(q_embed), k)
#     return "\n---\n".join([chunks[i] for i in I[0]])

# # Generate answer from FLAN-T5
# def generate_response(context, query):
#     prompt = f"""You are a smart data analyst assistant. Use the dataset context below to answer the user's question clearly and logically.

# If the data suggests a trend, explain it briefly. If there's not enough data, say so honestly.

# Context:
# {context}

# Question: {query}

# Answer:"""
#     return qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"].strip()






# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# # Load models
# qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load and preprocess CSV into chunks
# def load_chunks(csv_path):
#     df = pd.read_csv(csv_path).fillna("NA")
#     rows = [
#         "\n".join([f"{col}: {row[col]}" for col in df.columns])
#         for _, row in df.iterrows()
#     ]
#     # Domain-specific knowledge to improve answers
#     custom_chunks = [
#         "Loan_Status is the target variable. It is 'Y' for approved and 'N' for rejected loans.",
#         "Married applicants are slightly more likely to get loans, possibly due to combined incomes or coapplicants.",
#         "Unmarried applicants with good income and credit history can still get loans approved.",
#         "Credit_History is one of the strongest indicators of loan approval in the dataset.",
#         "ApplicantIncome and CoapplicantIncome together affect loan eligibility.",
#         "Education and Property_Area may also impact loan decisions slightly.",
#     ]
#     return custom_chunks + rows

# # Build FAISS index
# def build_index(chunks):
#     embeddings = model.encode(chunks)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings))
#     return index, chunks

# # Retrieve top relevant chunks
# def retrieve(query, index, chunks, k=3):
#     if not query or len(query.strip()) < 3:
#         return []
#     q_embed = model.encode([query])
#     D, I = index.search(np.array(q_embed), k)
#     return [chunks[i] for i in I[0] if i < len(chunks)]

# # Generate AI answer
# def generate_response(context, query):
#     if not context:
#         return "🙁 Sorry, I couldn’t find relevant information for your question."

#     context_text = "\n---\n".join(context)
#     prompt = f"""You are a smart loan approval assistant. 
# Use the dataset context below to answer the user's question clearly and logically.

# If the data suggests a trend, explain it briefly. 
# If there's not enough data, say so honestly.

# Context:
# {context_text}

# Question: {query}

# Answer:"""

#     try:
#         return qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"].strip()
#     except Exception as e:
#         return f"⚠️ Model error: {e}"

# # Suggested questions (for guidance in UI)
# SUGGESTED_QUERIES = [
#     "What affects loan approval the most?",
#     "How does Credit_History impact loan decisions?",
#     "Does marital status affect loan approval?",
#     "What role does ApplicantIncome play in loan approval?",
#     "How does Education influence loan approval chances?",
# ]






# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# # ------------------------------
# # Models
# # ------------------------------
# qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # ------------------------------
# # Predefined Suggested Questions
# # ------------------------------
# SUGGESTED_QUESTIONS = [
#     "What factors affect loan approval the most?",
#     "Does credit history impact loan approval?",
#     "Do married people have higher chances of loan approval?",
#     "How does applicant income influence loan status?",
#     "Does education level matter in loan approval?",
#     "Which property area has the highest approval rate?",
#     "What is the average loan amount of approved applications?",
# ]

# # ------------------------------
# # Load CSV into Chunks
# # ------------------------------
# def load_chunks(csv_path):
#     df = pd.read_csv(csv_path).fillna("NA")
#     rows = ["\n".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()]

#     # Some knowledge-based hints
#     custom_chunks = [
#         "Loan_Status is the target variable. It is 'Y' for approved and 'N' for rejected loans.",
#         "Credit_History is the most important factor in loan approval.",
#         "ApplicantIncome and CoapplicantIncome together affect loan eligibility.",
#         "Married applicants are slightly more likely to get loans due to combined income.",
#         "Education and Property_Area also influence approval, but less strongly.",
#     ]
#     return custom_chunks + rows

# # ------------------------------
# # Build FAISS Index
# # ------------------------------
# def build_index(chunks):
#     embeddings = model.encode(chunks)
#     index = faiss.IndexFlatL2(embeddings.shape[1])
#     index.add(np.array(embeddings))
#     return index, chunks

# # ------------------------------
# # Retrieve Relevant Chunks
# # ------------------------------
# def retrieve(query, index, chunks, k=3):
#     q_embed = model.encode([query])
#     D, I = index.search(np.array(q_embed), k)
#     return "\n---\n".join([chunks[i] for i in I[0]])

# # ------------------------------
# # Generate Response
# # ------------------------------
# def generate_response(context, query):
#     if not context.strip():
#         return "I couldn’t find anything relevant in the dataset. Try rephrasing your question or ask about general loan approval trends."

#     prompt = f"""
#     You are a helpful AI assistant for loan approval dataset analysis.
#     Use the provided context to answer the question clearly. 
#     If the context is not enough, say that honestly and give a general suggestion if possible.

#     Context:
#     {context}

#     Question: {query}

#     Answer:"""

#     try:
#         response = qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"].strip()
#     except Exception as e:
#         response = f"⚠️ Sorry, I couldn’t generate an answer. (Error: {str(e)})"
#     return response

# # ------------------------------
# # Suggest Questions
# # ------------------------------
# def suggest_questions():
#     return SUGGESTED_QUESTIONS











import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FLAN-T5 pipeline
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")


# # ------------------------------
# # Predefined Suggested Questions
# # ------------------------------
SUGGESTED_QUESTIONS = [
    "What factors affect loan approval the most?",
    "Does credit history impact loan approval?",
    "Do married people have higher chances of loan approval?",
    "How does applicant income influence loan status?",
    "Does education level matter in loan approval?",
    "Which property area has the highest approval rate?",
    "What is the average loan amount of approved applications?",
]

# ============================
# Custom Dataset Context
# ============================

CUSTOM_CONTEXT = [
    # Dataset Description
    "Dataset Structure and Details:\n"
    "Objective: Predict whether a person will get a loan based on input variables.\n"
    "Columns:\n"
    "- Loan_ID: Unique loan application ID\n"
    "- Gender: Male/Female\n"
    "- Married: Yes/No\n"
    "- Dependents: 0, 1, 2, 3+\n"
    "- Education: Graduate/Not Graduate\n"
    "- Self_Employed: Yes/No\n"
    "- ApplicantIncome: Monthly/Annual income\n"
    "- CoapplicantIncome: Co-applicant's income\n"
    "- LoanAmount: Requested loan amount\n"
    "- Loan_Amount_Term: Term in months\n"
    "- Credit_History: 1.0 = Yes, 0.0 = No\n"
    "- Property_Area: Urban/Semiurban/Rural\n"
    "- Loan_Status: Y/N (target variable)",

    # Key Insights
    "Key Insights:\n"
    "- Credit History is the most important factor. Good history → high approval chance.\n"
    "- Higher income applicants (and co-applicant income) → higher approval rate.\n"
    "- Married applicants have higher approval chances.\n"
    "- Graduates get approved more often than non-graduates.\n"
    "- Semi-urban areas have higher approval rates than urban/rural.\n"
    "- Smaller loan amounts and shorter terms → higher approval chance.\n"
    "- Self-employed applicants may face closer scrutiny.",

    # Potential Questions
    "Suggested Questions Users Can Ask:\n"
    "- What is the most important factor for loan approval?\n"
    "- Does my income affect loan approval?\n"
    "- Does property location matter?\n"
    "- If I am self-employed, is it difficult to get a loan?\n"
    "- How does marital status affect approval?\n"
    "- Do dependents affect loan eligibility?"
]

# ============================
# Load and Preprocess CSV into Chunks
# ============================
def load_chunks(csv_path):
    try:
        df = pd.read_csv(csv_path).fillna("NA")
    except Exception as e:
        raise FileNotFoundError(f"Error loading CSV: {e}")

    rows = [
        "\n".join([f"{col}: {row[col]}" for col in df.columns])
        for _, row in df.iterrows()
    ]

    # Combine custom context + rows from dataset
    return CUSTOM_CONTEXT + rows

# ============================
# Build FAISS Index
# ============================
def build_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

# ============================
# Retrieve Top Relevant Chunks
# ============================
def retrieve(query, index, chunks, k=3):
    if not query or len(query.strip()) < 3:
        return []

    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# ============================
# Generate Answer from FLAN-T5
# ============================
def generate_response(context, query):
    if not context:
        return "🙁 Sorry, I couldn’t find relevant information for your question."

    context_text = "\n---\n".join(context)
    prompt = f"""You are an expert loan assistant. Use the dataset context below to answer the user's question clearly and logically.

If the data suggests a trend, explain it briefly. If there's not enough data, say so honestly.

Context:
{context_text}

Question: {query}

Answer:"""

    try:
        return qa_pipeline(prompt, max_new_tokens=256)[0]["generated_text"].strip()
    except Exception as e:
        return f"⚠️ Model error: {e}"
