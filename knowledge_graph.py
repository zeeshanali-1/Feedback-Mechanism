import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# File paths
pdf_files = [
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Data Science RoadMap.pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Knowledge Base[1].pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Knowledge Base-copy_6month.pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Roadmap.pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/All.pdf'
]

# Function to extract text from PDF files
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Preprocess PDFs
def preprocess_pdfs():
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return raw_text

if __name__ == "__main__":
    preprocess_pdfs()
    print("PDFs processed and embeddings saved.")
