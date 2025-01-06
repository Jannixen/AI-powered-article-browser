import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import FastEmbedEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Qdrant
from langfuse.callback import CallbackHandler
from qdrant_client import QdrantClient

# Initialize Flask app
app = Flask(__name__)

# Constants for Qdrant and LLM setup

load_dotenv()
QDRANT_KEY = os.getenv('QDRANT_KEY')
QDRANT_CLUSTER_URL = os.getenv('QDRANT_CLUSTER_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Qdrant client and vector store
client = QdrantClient(url=QDRANT_CLUSTER_URL, api_key=QDRANT_KEY)
vector_store = Qdrant(
    client=client,
    collection_name="bbc_news_articles",
    embeddings=FastEmbedEmbeddings(),
)

langfuse_handler = CallbackHandler(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# Initialize memory, LLM, and QA chain
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
retriever = vector_store.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    callbacks=[langfuse_handler]
)


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Endpoint to handle user questions. Questions should be related to articles.
    Expects JSON input with a "question" field.
    """
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        # Get the answer from the ConversationalRetrievalChain
        response = qa({"question": question})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")

        # Return the answer as JSON
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    """
    Home route for testing the API.
    """
    return "BBC Articles QA Chatbot is running. Use the /ask endpoint to ask questions."


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# curl -X POST http://127.0.0.1:5000/ask -H "Content-Type: application/json" -d '{"question": "How much ipod with one gigabyte storage costs that was unveiled by Steve Jobs during annual MacWorld speech?"}'
