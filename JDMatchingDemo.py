import streamlit as st
from dotenv import load_dotenv
import os
import re
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("JD Matching With Resume Demo")

class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def load_and_clean_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}...")
            text = extract_text_from_pdf(file_path)
            clean_doc = clean_text(text)
            metadata = {"source": filename}
            documents.append(Document(page_content=clean_doc, metadata=metadata))
    return documents

# Directory containing PDF files
pdf_directory = "/home/mirafra/My_Project/RAG_demo/data"

# Load and clean PDF files
all_docs = load_and_clean_pdfs(pdf_directory)

# Split documents into chunks
try:
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    print("Documents split successfully.")
except Exception as e:
    st.error(f"Failed to split documents: {e}")
    st.stop()

# Debug print for documents
for doc in docs:
    print(doc.page_content)

# Generate embeddings and create a vector store
try:
    print("Generating embeddings and creating vector store...")
    embedding_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(docs, embedding_model)
    print("Vector store created successfully.")
except Exception as e:
    st.error(f"Failed to create vector store: {e}")
    st.stop()

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Initialize the Generative Model
llm = Ollama(model="qwen2:1.5b")

query = st.chat_input("Enter Job Description: ")

# Define the system prompt template
system_prompt = (
    "You are a helpful AI assistant with advanced knowledge in technology. "
    "Your role is to meticulously evaluate a candidate's resume based on the provided job description. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Provide the evaluation in the following format:\n\n"
    "Matching percentage: [matching_percentage]\n"
    "Matching Skills: [matching_skills]\n"
    "Missing Skills: [missing_skills]\n"
    "Resume File: [resume_file]\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

def calculate_matching_percentage(jd, resume):
    jd_skills = set(jd.lower().split())
    resume_skills = set(resume.lower().split())
    matching_skills = jd_skills.intersection(resume_skills)
    missing_skills = jd_skills - resume_skills
    matching_percentage = (len(matching_skills) / len(jd_skills)) * 100 if jd_skills else 0
    return matching_percentage, matching_skills, missing_skills

if query:
    try:
        # Retrieve relevant resumes
        retrieved_docs = retriever.get_relevant_documents(query)

        results = []
        for doc in retrieved_docs:
            resume_text = doc.page_content
            resume_file = doc.metadata.get("source", "Unknown")

            # Calculate matching percentage, matching skills, and missing skills
            matching_percentage, matching_skills, missing_skills = calculate_matching_percentage(query, resume_text)

            result = {
                "matching_percentage": matching_percentage,
                "matching_skills": list(matching_skills),
                "missing_skills": list(missing_skills),
                "resume_file": resume_file
            }
            results.append(result)

        # Display the results
        for result in results:
            st.write(f"Matching percentage: {result['matching_percentage']}%")
            st.write(f"Matching Skills: {result['matching_skills']}")
            st.write(f"Missing Skills: {result['missing_skills']}")
            st.write(f"Resume File: {result['resume_file']}")
            st.write("---")

    except Exception as e:
        st.error(f"Failed to generate response: {e}")
