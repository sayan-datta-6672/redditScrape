from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import requests

api_key = 'bHwsiXhQcJqVFkugcp4ysivj2K6Wa7NQ'
model = 'open-mixtral-8x22b-2404'
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

f = open('essay.txt', 'w')
f.write(text)
f.close()

# Load data
loader = TextLoader('essay.txt')
docs = loader.load()
# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# Define the embedding model
embeddings = MistralAIEmbeddings(model=model, mistral_api_key=api_key)
# Create the vector store
vector = FAISS.from_documents(documents, embeddings)
# Define a retriever interface
retriever = vector.as_retriever()
# Define LLM
model = ChatMistralAI(mistral_api_key=api_key)
# Define prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "What were the two main things the author worked on before college?"})
print(response["answer"])