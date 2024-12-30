import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

groq_api_key = os.getenv('GROQ_KEY')
os.environ["GEMINI_KEY"] = os.getenv("GEMINI_KEY")

st.title("Amine Karmous RAG Chatbot with Pdf")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
""")

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.docs = []

        
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFDirectoryLoader("temp")
            st.session_state.docs.extend(loader.load())

        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


os.makedirs("temp", exist_ok=True)


if "history" not in st.session_state:
    st.session_state.history = []


uploaded_files = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type=["pdf"])


if st.button("Process PDFs"):
    if uploaded_files:
        vector_embedding(uploaded_files)
        st.write("Vector Store DB is Ready")
    else:
        st.warning("Please upload at least one PDF file to proceed.")


prompt1 = st.text_input("Enter Your Question From Documents")


if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        end = time.process_time()
        st.write(f"Response time: {end - start:.4f} seconds")

        st.session_state.history.append({"question": prompt1, "answer": response['answer']})

        for chat in st.session_state.history:
            st.chat_message("user").markdown(chat['question'])
            st.chat_message("assistant").markdown(chat['answer'])

        st.write("---")
        with st.expander("Chat History"):
            for chat in st.session_state.history:
                st.write(f"**User:** {chat['question']}")
                st.write(f"**Assistant:** {chat['answer']}")
                st.write("--------------------------------")
    else:
        st.warning("Please process the PDFs first by clicking 'Process PDFs'.")

