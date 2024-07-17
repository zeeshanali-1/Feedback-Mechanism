import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "***"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"

# Initialize Neo4j graph
graph = Neo4jGraph()

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

# Paths to the PDF files
file_paths = [
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Data Science RoadMap.pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Knowledge Base[1].pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Knowledge Base-copy_6month.pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Roadmap.pdf'
]

# Extract text from each PDF
pdf_texts = [extract_text_from_pdf(file_path) for file_path in file_paths]

# Initialize LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
llm_transformer = LLMGraphTransformer(llm=llm)

# Create documents from the extracted text
documents = [Document(page_content=text) for text in pdf_texts]

# Convert documents to graph documents
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Print nodes and relationships
for idx, graph_doc in enumerate(graph_documents):
    #Unpack the tuple if necessary
    print(f"Document {idx+1}")
    print(f"Nodes: {graph_doc.nodes}")
    print(f"Relationships: {graph_doc.relationships}")
    graph.add_graph_documents(graph_doc)
    print()
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "sk-proj-IWDgMh2mEHPY0DYhUUocT3BlbkFJweuI2qvkUvco5kDxTTsc"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

# Initialize Neo4j graph
graph = Neo4jGraph()

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

# Paths to the PDF files
file_paths = [
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Data Science RoadMap.pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Knowledge Base[1].pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Knowledge Base-copy_6month.pdf',
    'C:/Users/kilar/OneDrive/Desktop/RAG/KB/Roadmap.pdf'
]

# Extract text from each PDF
pdf_texts = [extract_text_from_pdf(file_path) for file_path in file_paths]

# Initialize LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
llm_transformer = LLMGraphTransformer(llm=llm)

# Create documents from the extracted text
documents = [Document(page_content=text) for text in pdf_texts]

# Convert documents to graph documents
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Print nodes and relationships
for idx, graph_doc in enumerate(graph_documents):
    print(f"Document {idx+1}")
    print(f"Nodes: {graph_doc.nodes}")
    print(f"Relationships: {graph_doc.relationships}")
    
    print()

graph.add_graph_documents(graph_doc)