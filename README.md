import os
import streamlit as st
import pickle
import time
from pdfminer.high_level import extract_text
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Streamlit UI
st.title("Task-1: Chat with PDF Using RAG Pipeline")
st.sidebar.title("PDF Scraper")

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
process_pdf_clicked = st.sidebar.button("Process PDFs")
file_path = "faiss_store_openai.pkl"

# Placeholder for process messages
main_placeholder = st.empty()

# Process PDFs and create embeddings
if process_pdf_clicked:
    if uploaded_files:
        st.sidebar.success("Processing PDFs...")
        all_text = ""

        # Extract text from uploaded PDFs
        for uploaded_file in uploaded_files:
            extracted_text = extract_text(uploaded_file)
            all_text += extracted_text + "\n"

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_text(all_text)

        # Create embeddings and store in FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)

        # Save FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.sidebar.success("Vector Store Created Successfully!")
    else:
        st.sidebar.error("Please upload at least one PDF file.")

# Load vector store and handle queries
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, openai_api_key="your_openai_api_key")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Query input
    query = main_placeholder.text_input("Ask a Question about your PDFs:")

    if query:
        with st.spinner("Generating answer..."):
            result = chain.run(query)
        main_placeholder.write("*Answer:*")
        main_placeholder.write(result)
