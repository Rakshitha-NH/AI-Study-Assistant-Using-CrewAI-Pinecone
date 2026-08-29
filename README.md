# AI-Powered Study Assistant

An intelligent study assistant that allows users to upload study materials and ask questions. The system uses multiple specialized AI agents to retrieve relevant information and generate reliable answers.

## Features

- **Document Upload**: Upload PDF and text documents
- **Semantic Search**: Search documents using AI-powered embeddings
- **Multi-Agent System**: 
  - Research Agent: Retrieves relevant information
  - Analysis Agent: Analyzes and synthesizes information
  - Review Agent: Validates answer accuracy
- **Vector Database**: Pinecone for efficient semantic search
- **RESTful API**: Easy-to-use Flask API

## Architecture

```
User uploads PDF/Document
        ↓
Extract document text
        ↓
Generate embeddings (Sentence Transformers)
        ↓
Store in Pinecone Vector Database
        ↓
User asks question
        ↓
Search similar documents
        ↓
CrewAI Agents Process:
  - Research Agent (retrieval)
  - Analysis Agent (synthesis)
  - Review Agent (validation)
        ↓
Return verified answer
```

## Prerequisites

- Python 3.8+
- Ollama (for local LLM)
- Pinecone account (for vector storage)
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd study_assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Ensure Ollama is running**
   ```bash
   ollama serve
   # In another terminal, pull model
   ollama pull mistral
   ```

## Usage

### Start the Application
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

file: <binary-pdf-or-txt>
```

Response:
```json
{
  "status": "success",
  "document_id": "document_name",
  "chunks": 45,
  "message": "Document uploaded and processed successfully"
}
```

#### 3. Ask Question
```bash
POST /ask
Content-Type: application/json

{
  "question": "What is the main topic discussed in the document?"
}
```

Response:
```json
{
  "status": "success",
  "question": "What is the main topic?",
  "answer": "The document discusses...",
  "retrieved_documents": [
    {
      "id": "doc_0",
      "score": 0.92,
      "text": "Relevant text chunk...",
      "document_id": "document_name"
    }
  ],
  "document_count": 5
}
```

#### 4. Search Documents
```bash
POST /search
Content-Type: application/json

{
  "query": "topic to search",
  "top_k": 5
}
```

#### 5. List Documents
```bash
GET /documents
```

#### 6. Delete Document
```bash
DELETE /documents/<document_id>
```

## Project Structure

```
study_assistant/
├── app.py                    # Flask application
├── agent_crew.py             # CrewAI agents and crew
├── config.py                 # Configuration settings
├── document_processor.py      # Document extraction and chunking
├── vector_store.py           # Pinecone integration
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── uploads/                  # Uploaded documents folder
└── README.md                 # This file
```

## Technologies Used

- **Flask**: Web framework for API
- **CrewAI**: Agent orchestration framework
- **LangChain**: LLM integration
- **Pinecone**: Vector database for semantic search
- **Sentence Transformers**: Text embedding model
- **Ollama**: Local LLM runtime
- **PyPDF2**: PDF processing

## Configuration

Edit `.env` file to customize:

```env
# Pinecone
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-east1-aws
PINECONE_INDEX_NAME=study-assistant

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Flask
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
```

## Workflow

1. User uploads a document (PDF/TXT)
2. System extracts text and splits into chunks
3. Embeddings are generated using Sentence Transformers
4. Embeddings are stored in Pinecone
5. User asks a question
6. System retrieves top-k similar documents
7. CrewAI agents process the query:
   - Research Agent: Extracts relevant information
   - Analysis Agent: Creates comprehensive answer
   - Review Agent: Validates the answer
8. Final answer is returned to user

## Error Handling

The application includes comprehensive error handling for:
- File upload errors
- Document processing failures
- Pinecone connectivity issues
- LLM processing errors
- Invalid requests

## Performance Optimization

- Chunking strategy optimizes embedding quality
- Cosine similarity search for fast retrieval
- Batch processing for multiple queries
- Connection pooling for Pinecone

## Future Enhancements

- [ ] Support for more document formats (DOCX, PPT)
- [ ] Web interface with HTML/CSS
- [ ] User authentication and session management
- [ ] Document categorization and tagging
- [ ] Answer confidence scoring
- [ ] Chat history and context management
- [ ] Multi-language support

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request
