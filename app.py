import streamlit as st
from bs4 import BeautifulSoup
import requests
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from transformers import pipeline

# Initialize models
@st.cache_resource
def load_models():
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return retriever, generator

retriever_model, generator = load_models()

# Load index and chunks
@st.cache_resource
def load_data():
    index = faiss.read_index("pdf_qa.index")
    with open("pdf_chunks.json", "r") as f:
        chunks = json.load(f)
    return index, chunks

index, chunks = load_data()

# Retrieve relevant chunks
def get_relevant_chunks(question, k=3):
    q_embedding = retriever_model.encode([question])
    D, I = index.search(np.array(q_embedding), k=k)
    return [(chunks[idx], float(D[0][i])) for i, idx in enumerate(I[0])]

# Generate an answer
def generate_answer(question, context):
    input_text = f"""Answer the following question based on the given context.
    Be concise and accurate. If you don't know, say so.

    Question: {question}
    Context: {context}

    Answer:"""
    
    result = generator(input_text, max_length=200, temperature=0.7)
    return result[0]['generated_text']

# Full pipeline
def answer_question(question):
    relevant_chunks_with_scores = get_relevant_chunks(question)
    relevant_chunks = [chunk for chunk, score in relevant_chunks_with_scores]
    scores = [score for chunk, score in relevant_chunks_with_scores]
    
    context = " ".join(relevant_chunks)
    answer = generate_answer(question, context)
    
    confidence = 1 - (sum(scores) / len(scores)) if scores else 0
    
    return {
        "question": question,
        "answer": answer,
        "confidence": min(max(confidence, 0), 1),
        "sources": relevant_chunks
    }

# Streamlit UI
st.title("Website ChatBot of Software company")

st.markdown("Ask any question ")

question_input = st.text_input("Your Question", placeholder="Ask about this Documents ")

if st.button("Get Answer") and question_input:
    with st.spinner("Processing..."):
        response = answer_question(question_input)

    st.markdown(f"### ✅ Answer (Confidence: `{response['confidence']:.0%}`)")
    st.markdown("""
    <style>
    .footer-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 30px;
        background-color: transparent;
        overflow: hidden;
        z-index: 100;
    }

    .scrolling-text {
        display: inline-block;
        padding-left: 100%;
        animation: scroll-left-to-right 15s linear infinite;
        white-space: nowrap;
        color: #4CAF50;
        font-size: 22px;
        font-weight: bold;
    }

    @keyframes scroll-left-to-right {
        0%   { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    </style>

    <div class="footer-container">
        <div class="scrolling-text">Developed by Md Johurul Islam</div>
    </div>
   """, unsafe_allow_html=True)

    
    st.success(response['answer'])

    if st.checkbox("🔍 Show Sources"):
        for i, src in enumerate(response["sources"], 1):
            st.markdown(f"**Source {i}:**")
            st.code(src[:1000] + ("..." if len(src) > 1000 else ""))
