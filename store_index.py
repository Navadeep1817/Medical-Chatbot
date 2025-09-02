import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Import helper functions
from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)

# Load environment variables from .env
load_dotenv()

# Load and process documents
data_path = "data"  # Change this if PDFs are stored elsewhere
docs = load_pdf_files(data_path)
minimal_docs = filter_to_minimal_docs(docs)
chunks = text_split(minimal_docs)

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "medical-chatbot"

# Create index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # all-MiniLM-L6-v2 embeddings dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

# Connect to Pinecone index
index = pc.Index(index_name)

# Store vectors in Pinecone using LangChain wrapper
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print(f" Successfully stored {len(chunks)} chunks into Pinecone index '{index_name}'.")
