import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set up Streamlit page configuration
st.set_page_config(page_title="Document-Based Inquiry with Llama3", layout="wide")
st.title("Explore PDF Documents Using the Llama3 Model")
st.write("Hello! I'm your friendly LLM chatbot. Created By Jillani SoftTech 😎")
st.markdown("""
    Welcome to our document exploration tool. You can interact with a corpus of PDF documents by typing your questions below. Our tool utilizes the Llama3 model to provide precise responses based on the content of these documents.
""")

def initialize_session_state():
    """Initializes the session state for document processing and embedding."""
    if "initialized" not in st.session_state:
        try:
            st.session_state.embeddings = OllamaEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("epidemiologyBooks/")
            st.session_state.docs = st.session_state.loader.load()
            if not st.session_state.docs:
                st.error("No documents found. Please verify the directory path.")
                return

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.processed_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vector_store = FAISS.from_documents(st.session_state.processed_documents, st.session_state.embeddings)
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Error during session initialization: {e}")
            return

def initialize_llm():
    """Initializes the Llama3 language model with the Groq API key."""
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.error("Groq API key is missing. Ensure it's set in your .env file.")
        return None
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192",temperature=0.2)

llm = initialize_llm()
initialize_session_state()

if llm and st.session_state.get("initialized"):
    prompt_template = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Provide the most accurate response based on the question:
        <context>
        {context}
        <context>
        Questions: {input}
    """)

    combined_document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combined_document_chain)

    user_query = st.text_input("Enter your question here:")
    if user_query:
        start = time.process_time()
        try:
            response = retrieval_chain.invoke({"input": user_query})
            response_time = time.process_time() - start
            st.write(f"Response processed in {response_time:.2f} seconds.")
            st.write(response['answer'])
            with st.expander("View Similar Document Snippets"):
                for i, doc in enumerate(response["context"]):
                    st.write(f"Document {i + 1}:")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            feedback = st.radio("Was this answer helpful?", ('Yes', 'No'))
            if feedback:
                st.session_state.feedback = feedback
                if feedback == 'No':
                    st.text_area("Please provide more details on how we can improve:", key='feedback_details')
        except Exception as e:
            st.error(f"Error during response retrieval: {e}")
else:
    st.warning("LLM initialization failed or documents are not loaded. Please verify the API key and document directory.")
