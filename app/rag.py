from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# Initialize embeddings (Sentence Transformers)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store_path = "faiss_index"

def get_vector_store():
    if os.path.exists(vector_store_path):
        return FAISS.load_local(vector_store_path, embeddings)
    return None

def ingest_document(file_path: str):
    # For simplicity, assuming text files
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Generate embeddings and store
    vector_store = get_vector_store()
    if vector_store is None:
        vector_store = FAISS.from_documents(texts, embeddings)
    else:
        vector_store.add_documents(texts)
    
    vector_store.save_local(vector_store_path)
    return len(texts)

def answer_question(question: str):
    vector_store = get_vector_store()
    if not vector_store:
        return "No documents ingested yet. Please upload a document first."
    
    # Retrieve relevant context
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # In a real app, we would pass this context to an LLM (OpenAI, local Llama, etc.)
    # Mocking LLM response generation based on context for demonstration
    if not context:
        return "I could not find relevant context to answer your question."
        
    return f"Based on the provided documents, here is the relevant context:\n{context}\n\n(LLM generation would synthesize this context into a final answer)."

