# from transformers import pipeline
# from utils import load_loan_documents, build_faiss_index, search_index
# import os

# CSV_PATH = "data/Training.csv"

# if not os.path.exists("embeddings/faiss_index.bin"):
#     print("Index not found. Building FAISS index...")
#     docs = load_loan_documents(CSV_PATH)
#     build_faiss_index(docs)
#     print("Index built.")

# print("Loading FLAN-T5 model...")
# generator = pipeline("text2text-generation", model="google/flan-t5-base")

# def rag_chatbot(query):
    
#     if len(query.strip()) < 3:
#         return "Please ask a more specific question about loan applications."

#     relevant_docs = search_index(query)
#     context = "\n".join(f"- {doc}" for doc in relevant_docs)
    
#     prompt = (
#         f"You are an expert loan assistant.\n"
#         f"Use the information below to answer the question clearly and accurately.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question: {query}\nAnswer:"
#     )

#     result = generator(prompt, max_new_tokens=256)[0]["generated_text"]
#     return result

# if __name__ == "__main__":
#     print("RAG Chatbot ready. Type your loan-related questions!")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Bot: Goodbye!")
#             break
#         answer = rag_chatbot(user_input)
#         print(f"Bot: {answer}\n")









# import streamlit as st
# from utils import load_chunks, build_index, retrieve, generate_response

# # 🛠 Page Setup
# st.set_page_config(page_title="Loan Approval RAG Chatbot", layout="wide", page_icon="📊")

# # 🌙 Dark Theme Styling
# st.markdown("""
#     <style>
#         body {
#             background-color: #1e1e2f;
#             color: #f0f0f0;
#         }
#         .stApp {
#             background-color: #1e1e2f;
#         }
#         h1 {
#             text-align: center;
#             color: #f8f8ff;
#             font-weight: bold;
#         }
#         .subtitle {
#             text-align: center;
#             color: #cccccc;
#             font-size: 18px;
#             margin-top: 0px;
#         }
#         .stTextInput input {
#             background-color: #2d2d44;
#             color: #ffffff;
#             border: 2px solid #4e8cff;
#             border-radius: 10px;
#             padding: 0.6rem;
#             font-size: 18px;
#         }
#         .stButton>button {
#             background-color: #4e8cff;
#             color: white;
#             border-radius: 8px;
#             font-size: 16px;
#             padding: 0.5rem 1rem;
#             border: none;
#         }
#         .stButton>button:hover {
#             background-color: #3b6edb;
#             transition: 0.3s;
#         }
#         .block-container {
#             padding-top: 2rem;
#         }
#         .footer {
#             text-align: center;
#             color: #888;
#             margin-top: 3rem;
#             font-size: 14px;
#         }
#         .context-box {
#             background-color: #2a2a3d;
#             color: #f0f0f0;
#             padding: 1rem;
#             border-radius: 10px;
#             box-shadow: 0 0 10px rgba(255,255,255,0.1);
#             font-family: monospace;
#             font-size: 14px;
#             white-space: pre-wrap;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # 🔹 Header
# st.markdown("<h1>📊 Loan Approval RAG Chatbot</h1>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Ask questions from the dataset using retrieval-augmented generation</div>", unsafe_allow_html=True)

# # 🧠 Load Embeddings
# @st.cache_resource
# def setup():
#     chunks = load_chunks("data/Training.csv")
#     index, all_chunks = build_index(chunks)
#     return index, all_chunks

# index, all_chunks = setup()

# # 💬 Input Section (no random button)
# st.markdown("### 💬 Ask your question:")

# if "user_query" not in st.session_state:
#     st.session_state.user_query = ""

# query = st.text_input(
#     "",
#     value=st.session_state.user_query,
#     placeholder="e.g. What affects loan approval the most?",
#     label_visibility="collapsed",
#     key="user_query"
# )

# # 🤖 AI Response
# if query:
#     with st.spinner("🔍 Generating response..."):
#         context = retrieve(query, index, all_chunks)
#         answer = generate_response(context, query)

#     st.markdown("### 💡 AI Answer")
#     st.success(answer)

#     with st.expander("📄 View Retrieved Context"):
#         st.markdown(f"<div class='context-box'>{context}</div>", unsafe_allow_html=True)

# # 🧾 Footer
# st.markdown("<div class='footer'>Built with ❤️ using Streamlit, FAISS & FLAN-T5</div>", unsafe_allow_html=True)














import streamlit as st
from utils import load_chunks, build_index, retrieve, generate_response, SUGGESTED_QUESTIONS

# 🛠 Page Setup
st.set_page_config(page_title="Loan Approval RAG Chatbot", layout="wide", page_icon="📊")

# 🌙 Custom Styling
st.markdown("""
    <style>
        .stApp { background-color: #1e1e2f; }
        h1 { text-align: center; color: #f8f8ff; font-weight: bold; }
        .subtitle { text-align: center; color: #cccccc; font-size: 18px; margin-top: 0px; }
        .stTextInput input {
            background-color: #2d2d44; color: #ffffff;
            border: 2px solid #4e8cff; border-radius: 10px;
            padding: 0.6rem; font-size: 18px;
        }
        .stButton>button {
            background-color: #4e8cff; color: white; border-radius: 8px;
            font-size: 16px; padding: 0.5rem 1rem; border: none;
        }
        .stButton>button:hover { background-color: #3b6edb; transition: 0.3s; }
        .context-box {
            background-color: #2a2a3d; color: #f0f0f0;
            padding: 1rem; border-radius: 10px;
            box-shadow: 0 0 10px rgba(255,255,255,0.1);
            font-family: monospace; font-size: 14px; white-space: pre-wrap;
        }
        .footer { text-align: center; color: #888; margin-top: 3rem; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# 🔹 Header
st.markdown("<h1>📊 Loan Approval RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions from the dataset using retrieval-augmented generation</div>", unsafe_allow_html=True)

# 🧠 Load Data & Build Index
@st.cache_resource
def setup():
    chunks = load_chunks("data/Training.csv")
    index, all_chunks = build_index(chunks)
    return index, all_chunks

index, all_chunks = setup()

# 💬 Input Section
st.markdown("### 💬 Ask your question:")

query = st.text_input(
    "",
    placeholder="e.g. What affects loan approval the most?",
    label_visibility="collapsed"
)

# 🤖 AI Response
if query:
    with st.spinner("🔍 Generating response..."):
        context = retrieve(query, index, all_chunks)
        answer = generate_response(context, query)

    st.markdown("### 💡 AI Answer")
    st.success(answer)

    with st.expander("📄 View Retrieved Context"):
        st.markdown(f"<div class='context-box'>{'---'.join(context)}</div>", unsafe_allow_html=True)

# 💡 Suggested Questions
st.markdown("### 💡 Suggested Questions You Can Ask")
cols = st.columns(2)
for i, q in enumerate(SUGGESTED_QUESTIONS):
    if cols[i % 2].button(q):
        st.session_state.user_query = q
        st.experimental_rerun()

# 🧾 Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit, FAISS & FLAN-T5</div>", unsafe_allow_html=True)
