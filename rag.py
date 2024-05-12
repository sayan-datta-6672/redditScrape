import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader

st.title("Chat with webpage")
st.caption("")

webpage_url = st.text_input("Enter web page URL: ", type="default")
model_name = "mixtral:8x7b"

if webpage_url:
    # 1. load the data
    loader = WebBaseLoader(webpage_url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    # 2. create ollama embeddings and vector store
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 3. call ollama llama3 model
    def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']

    # 4. rag setup
    retriever = vectorstore.as_retriever()

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_context = combine_docs(retrieved_docs)
        return ollama_llm(question, formatted_context)

    st.success(f"Loaded {webpage_url} successfully.")

    # ask a question about the webpage
    prompt = st.text_input("Ask any question about the webpage")

    # chat with the webpage
    if prompt:
        result = rag_chain(prompt)
        st.write(result)
    
