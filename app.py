import streamlit as st
import zipfile
import os
import pickle
import pandas as pd

import tensorflow as tf
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain.chains import load_summarize_chain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====================
# Chroma DB Extraction
# ====================
CHROMA_DIR = "chroma_db"

if not os.path.exists(CHROMA_DIR):
    with zipfile.ZipFile("chroma_db.zip", "r") as zip_ref:
        zip_ref.extractall(CHROMA_DIR)

# Load Chroma DB
vector_store = Chroma(persist_directory=CHROMA_DIR)

# Load serialized models
with open("embeddings_model.pkl", "rb") as f:
    embeddings_model = pickle.load(f)

with open("llm_model.pkl", "rb") as f:
    llm = pickle.load(f)

with open("qa_chain_llm.pkl", "rb") as f:
    qa_chain = pickle.load(f)

with open("summarization_chain.pkl", "rb") as f:
    summarization_chain = pickle.load(f)


# ======================
# Helper Functions
# ======================

def custom_retriever(query, vector_store, make, model, year):
    all_docs = vector_store._collection.get()

    filtered_docs = [
        (doc, all_docs['documents'][index])
        for index, doc in enumerate(all_docs['metadatas'])
        if doc.get("make") == make and doc.get("model") == model and doc.get("year") == year
    ]

    return filtered_docs


def get_most_relevant_metadata(response, filtered_docs):
    doc_contents = [doc_content for _, doc_content in filtered_docs]
    vectorizer = TfidfVectorizer()
    all_texts = [response] + doc_contents
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    highest_similarity_idx = similarity_scores.argmax()
    most_relevant_metadata = filtered_docs[highest_similarity_idx][0]
    return most_relevant_metadata


def process_query(input_data, vector_store, qa_chain, llm):
    make_filter = input_data["make"]
    model_filter = input_data["model"]
    year_filter = int(input_data["year"])
    issue_query = input_data["issue"] + " risk in detail"

    filtered_docs = custom_retriever(issue_query, vector_store, make_filter, model_filter, year_filter)

    if not filtered_docs:
        return "No relevant documents found.", None, None

    metadata_list = [metadata for metadata, _ in filtered_docs]
    doc_contents = [doc_content for _, doc_content in filtered_docs]

    qa_chain_input_docs = [Document(page_content=content) for content in doc_contents]
    response = qa_chain.run(input_documents=qa_chain_input_docs, query=issue_query)
    relevant_metadata = get_most_relevant_metadata(response, filtered_docs)

    summary = summarization_chain({"input_documents": [Document(page_content=response)]})

    return response, relevant_metadata, summary['output_text']


# ====================
# Streamlit UI
# ====================

st.title("🚗 Car Issue RAG Summarizer")

make = st.text_input("Enter Car Make")
model = st.text_input("Enter Car Model")
year = st.text_input("Enter Car Year")
issue = st.text_area("Describe the Issue")

if st.button("Generate Summary"):
    if not (make and model and year and issue):
        st.error("Please fill all fields.")
    else:
        input_data = {
            "make": make,
            "model": model,
            "year": year,
            "issue": issue
        }

        with st.spinner("Processing..."):
            response, metadata, summary = process_query(input_data, vector_store, qa_chain, llm)

        if metadata is None:
            st.warning("No relevant documents found.")
        else:
            st.subheader("📄 Most Relevant Metadata")
            st.json(metadata)

            st.subheader("🤖 QA Chain Response")
            st.write(response)

            st.subheader("📝 Summary")
            st.write(summary)
