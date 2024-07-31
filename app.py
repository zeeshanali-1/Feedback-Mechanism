import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# File path for the student performance data
excel_file_path = 'C:/Users/kilar/deepu/student_performance.csv'

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
    Create a script for a 60-second video feedback for the student named {user_data['user_name']} based on the following performance data:
    Machine Learning Course Completion: {user_data['ML_completion']}%, 
    Deep Learning Course Completion: {user_data['DL_completion']}%,
    Quantitative Aptitude Course Completion: {user_data['Quant_completion']}%, 
    SQL Course Completion: {user_data['SQL_completion']}%,
    Python Course Completion: {user_data['Python_completion']}%,
    Number of Weekly Tests: {user_data['weekly_tests']},
    Average Weekly Test Marks: {user_data['weekly_test_marks']}%,
    Number of Sectional Tests: {user_data['sectional_tests']},
    Average Sectional Test Marks: {user_data['sectional_test_marks']}%.

    Current period: Month {user_data['month']}, Week {user_data['week']}

    Provide a comprehensive paragraph that contains an introduction which includes month and week, highlights of the student's performance, areas needing improvement, specific suggestions for improvement, and concluding remarks based on the month and week.
    """
    return user_input(feedback_question, knowledge_base)

# Main function to set up Streamlit interface
def main():
    st.set_page_config(page_title="Feedback Mechanism")
    st.header("Feedback Mechanism")

    st.header("Enter User ID")
    user_id = st.text_input("User ID")

    # Initialize a session state variable for knowledge_base
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = True  # Placeholder to indicate knowledge base is loaded

    if st.button("Generate Feedback"):
        with st.spinner("Generating feedback..."):
            # Read the Excel file into a DataFrame
            if excel_file_path.endswith(".csv"):
                df = pd.read_csv(excel_file_path)
            else:
                df = pd.read_excel(excel_file_path)

            # Search for the user ID in the DataFrame
            user_data = df[df["user_id"] == int(user_id)].to_dict(orient="records")
            
            if not user_data:
                st.warning("User ID not found in the uploaded file.")
            else:
                user_data = user_data[0]
                feedback = generate_feedback(user_data, st.session_state.knowledge_base)
                st.write("Feedback: ", feedback)

if __name__ == "__main__":
    main()
