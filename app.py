import streamlit as st
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

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
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
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, search in the Gemini model directly.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to search Gemini for the answer
def search_gemini(user_question):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
        response = llm.invoke(user_question)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle user input and get the response
def user_input(user_question, knowledge_base):
    if knowledge_base:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        context = "\n".join([doc.page_content for doc in docs])
        response = chain({"context": context, "question": user_question}, return_only_outputs=True)
        if "I cannot answer this question from the provided context" in response["output_text"]:
            gemini_response = search_gemini(user_question)
            response["output_text"] = gemini_response
        
        return response["output_text"]
    else:
        return search_gemini(user_question)

# Function to generate feedback based on user data and knowledge base
def generate_feedback(user_data, knowledge_base):
    feedback_question = f"""
    Generate feedback for a student with the following details:
    - User ID: {user_data['user_id']}
    - User Name: {user_data['user_name']}
    - Email: {user_data['email']}
    - Phone Number: {user_data['phone']}
    - ML Course Completion: {user_data['ml_completion']}%
    - DL Course Completion: {user_data['dl_completion']}%
    - Quant Course Completion: {user_data['quant_completion']}%
    - SQL Course Completion: {user_data['sql_completion']}%
    - Python Course Completion: {user_data['python_completion']}%
    - No. of Weekly Tests: {user_data['weekly_tests']}
    - Average Weekly Test Marks: {user_data['avg_weekly_marks']}%
    - No. of Sectional Tests: {user_data['sectional_tests']}
    - Average Sectional Marks: {user_data['avg_sectional_marks']}%

    Based on the knowledge base, please provide feedback on the following:
    1. Subjects where the student is excelling.
    2. Subjects where the student needs improvement.
    3. Suggestions to improve in the subjects where the student is lagging.
    """
    return user_input(feedback_question, knowledge_base)

# Main function to set up Streamlit interface
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini💁")
    st.header("Chat with PDF using Gemini💁")

    st.header("Enter User Details")
    user_id = st.text_input("User ID")
    user_name = st.text_input("User Name")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    ml_completion = st.slider("ML Course Completion (%)", 0, 100)
    dl_completion = st.slider("DL Course Completion (%)", 0, 100)
    quant_completion = st.slider("Quant Course Completion (%)", 0, 100)
    sql_completion = st.slider("SQL Course Completion (%)", 0, 100)
    python_completion = st.slider("Python Course Completion (%)", 0, 100)
    weekly_tests = st.number_input("No. of Weekly Tests", min_value=0)
    avg_weekly_marks = st.slider("Average Weekly Test Marks", 0, 100)
    sectional_tests = st.number_input("No. of Sectional Tests", min_value=0)
    avg_sectional_marks = st.slider("Average Sectional Marks", 0, 100)

    st.header("Upload PDF for Knowledge Base")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    
    knowledge_base = ""

    if st.button("Process PDF"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                knowledge_base = raw_text
                st.success("Knowledge base created successfully!")
                st.write("Knowledge Base Content:")
                st.write(knowledge_base)
        else:
            st.warning("Please upload at least one PDF file.")

    if st.button("Generate Feedback"):
        with st.spinner("Generating feedback..."):
            user_data = {
                "user_id": user_id,
                "user_name": user_name,
                "email": email,
                "phone": phone,
                "ml_completion": ml_completion,
                "dl_completion": dl_completion,
                "quant_completion": quant_completion,
                "sql_completion": sql_completion,
                "python_completion": python_completion,
                "weekly_tests": weekly_tests,
                "avg_weekly_marks": avg_weekly_marks,
                "sectional_tests": sectional_tests,
                "avg_sectional_marks": avg_sectional_marks
            }
            feedback = generate_feedback(user_data, knowledge_base)
            st.write("Feedback: ", feedback)

if __name__ == "__main__":
    main()
