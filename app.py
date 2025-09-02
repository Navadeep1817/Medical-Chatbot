from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize components
embeddings = None
docsearch = None
retriever = None
chatModel = None
rag_chain = None

try:
    logger.info("Loading embeddings...")
    embeddings = download_hugging_face_embeddings()
    logger.info("Embeddings loaded successfully")
except Exception as e:
    logger.error(f"Error loading embeddings: {str(e)}")

try:
    logger.info("Connecting to Pinecone...")
    index_name = "medical-chatbot" 
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logger.info("Pinecone connected successfully")
except Exception as e:
    logger.error(f"Error connecting to Pinecone: {str(e)}")

try:
    logger.info("Initializing Ollama with Mistral model...")
    # Use Mistral model directly
    chatModel = ChatOllama(model="mistral", temperature=0.7, timeout=60)
    # Test with a simple message
    test_response = chatModel.invoke("Hello")
    logger.info(f"Successfully connected to Ollama with model: mistral")
except Exception as e:
    logger.error(f"Error initializing Ollama with Mistral: {str(e)}")
    chatModel = None

# Set up RAG chain if all components are available
if all([embeddings, docsearch, chatModel]):
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        logger.info("RAG chain initialized successfully")
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
else:
    logger.warning("RAG chain not initialized due to missing components")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        if rag_chain is None:
            # Fallback: use Ollama directly without RAG
            if chatModel:
                msg = request.form["msg"]
                response = chatModel.invoke(msg)
                return response.content
            else:
                return "System not fully initialized. Please check server logs."
        
        msg = request.form["msg"]
        logger.info(f"Received message: {msg}")
        
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
        logger.info(f"Response: {answer}")
        
        return answer
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return "I'm having trouble processing your request. Please try again."

@app.route("/health")
def health_check():
    """Endpoint to check system health"""
    status = {
        "embeddings_loaded": embeddings is not None,
        "pinecone_connected": docsearch is not None,
        "ollama_connected": chatModel is not None,
        "rag_chain_ready": rag_chain is not None,
        "status": "healthy" if all([embeddings, docsearch, chatModel, rag_chain]) else "partial" if chatModel else "unhealthy"
    }
    return jsonify(status)

@app.route("/test_ollama")
def test_ollama():
    """Test Ollama directly"""
    try:
        if chatModel:
            response = chatModel.invoke("Hello, how are you?")
            return f"Ollama test successful: {response.content}"
        else:
            # Try to create a temporary connection
            test_model = ChatOllama(model="mistral", timeout=30)
            response = test_model.invoke("Hello")
            return f"Ollama test successful (temporary): {response.content}"
    except Exception as e:
        return f"Ollama test failed: {str(e)}. Make sure Ollama is installed and models are downloaded."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)