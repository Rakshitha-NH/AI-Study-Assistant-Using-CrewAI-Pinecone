import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import our optimized crew logic
from agent_crew import run_study_crew

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use env variables for security
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "study-assistant")

pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_index():
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(PINECONE_INDEX_NAME)

# --- PDF PROCESSING ---
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(docs)

def upsert_documents(chunks, filename):
    index = get_index()
    texts = [doc.page_content for doc in chunks]
    metadatas = [{"text": text, "source": filename} for text in texts]
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    embeds = embeddings_model.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))

# --- FAST SEARCH ---
def search_knowledge_base(query: str) -> str:
    index = get_index()
    query_vector = embeddings_model.embed_query(query)
    # Getting top 3 most relevant chunks
    result = index.query(vector=query_vector, top_k=3, include_metadata=True)
    contexts = [match['metadata']['text'] for match in result.get('matches', [])]
    return "\n\n".join(contexts) if contexts else "No relevant info found."

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        try:
            chunks = process_pdf(filepath)
            upsert_documents(chunks, filename)
            return jsonify({'message': 'Document indexed! You can now ask questions.'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    try:
        # STEP 1: Search Pinecone (Fast - few milliseconds)
        context = search_knowledge_base(question)
        
        # STEP 2: Run CrewAI with the found context (One LLM pass - much faster)
        answer = run_study_crew(question, context)
        
        return jsonify({'answer': str(answer)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': "The AI is taking too long to respond. Please try again."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)