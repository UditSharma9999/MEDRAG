# app.py

import streamlit as st
from rag_engine import process_pdfs, retrieve_context, get_gemini_response
import os

st.set_page_config(page_title="MEDRAG - Medical AI Assistant", layout="wide")
st.title("🩺 MEDRAG - Medical Document RAG Assistant")

DATA_FOLDER = r"data"

@st.cache_data
def process_data():
    process_pdfs(DATA_FOLDER)

process_data()

# # Sidebar: Initial Setup
# with st.sidebar:
#     st.header("📄 PDF Loader")
#     if st.button("Process PDFs"):
#         process_pdfs(DATA_FOLDER)
#         st.success("PDFs processed and stored in vector DB!")

#     st.markdown("---")
#     st.write("Ensure PDFs are in this path:")
#     st.code(DATA_FOLDER)


# Main Panel
query = st.text_input("🔍 Ask your medical question:")
if query:
    process_pdfs(DATA_FOLDER)
    print("dataaaaa")
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        chunks = retrieve_context(query)
        answer = get_gemini_response(query, chunks)

    st.subheader("💬 Gemini's Answer")
    st.markdown(answer)

    with st.expander("📄 Retrieved Chunks"):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.info(chunk)
