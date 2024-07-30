import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# File paths
pdf_paths = [
    'Data Science RoadMap.pdf',
    'Knowledge Base-copy-updated_weekly.pdf',
    'Knowledge Base-copy_6month.pdf',
    'Roadmap.pdf'
]
excel_file_path = 'student_performance.csv'

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

# Function to create a conversational chain for QA
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, acknowledge that the information is not available in the knowledge base.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and get the response
def user_input(user_question, knowledge_base):
    if knowledge_base:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        if not docs:
            return "No relevant information found in the knowledge base."

        context = "\n".join([doc.page_content for doc in docs])
        response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
        
        if "The provided context does not contain the information necessary to answer this question" in response["output_text"]:
            return "The knowledge base does not contain sufficient information to answer the question. Please ensure the PDFs contain relevant details."
        
        return response["output_text"]
    else:
        return "Knowledge base is empty. Please process the PDF files to create the knowledge base."

# Function to generate feedback based on user data and knowledge base
def generate_feedback(user_data, knowledge_base):
    feedback_question = f"""
    Considering the student's performance in the following courses and tests:
    - Machine Learning Course Completion: {user_data['ML_completion']}%
    - Deep Learning Course Completion: {user_data['DL_completion']}%
    - Quantitative Aptitude Course Completion: {user_data['Quant_completion']}%
    - SQL Course Completion: {user_data['SQL_completion']}%
    - Python Course Completion: {user_data['Python_completion']}%
    - Number of Weekly Tests: {user_data['weekly_tests']}
    - Average Weekly Test Marks: {user_data['weekly_test_marks']}%
    - Number of Sectional Tests: {user_data['sectional_tests']}
    - Average Sectional Test Marks: {user_data['sectional_test_marks']}%

    Please provide a detailed feedback covering the following aspects:
    1. Identify the subjects where the student is performing well.
    2. Highlight the subjects where the student needs improvement.
    3. Provide specific suggestions and strategies to help the student improve in the subjects where they are lagging.
    4. Suggest additional resources or study materials based on the student's current performance.
    5. Provide an overall summary of the student's performance.
    """
    return user_input(feedback_question, knowledge_base)

# Main function to set up Streamlit interface
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini💁")
    st.header("Chat with PDF using Gemini💁")

    st.header("Enter User ID")
    user_id = st.text_input("User ID")

    # Initialize a session state variable for knowledge_base
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = ""

    
    raw_text = get_pdf_text(pdf_paths)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    st.session_state.knowledge_base = raw_text  # Update session state
    

    
    if st.button("Generate Feedback"):
        with st.spinner("Generating feedback..."):
            # Read the Excel file into a DataFrame
            if excel_file_path.endswith(".csv"):
                df = pd.read_csv(excel_file_path)
            else:
                df = pd.read_excel(excel_file_path)

            # Search for the user ID in the DataFrame
            if user_id.isdigit():
                user_data = df[df["user_id"] == int(user_id)].to_dict(orient="records")
            else:
                st.warning("Please enter a valid User ID.")
                return

            if not user_data:
                st.warning("User ID not found in the uploaded file.")
            else:
                user_data = user_data[0]
                feedback = generate_feedback(user_data, st.session_state.knowledge_base)  # Use session state
                st.write("Feedback: ", feedback)

if __name__ == "__main__":
    main()



