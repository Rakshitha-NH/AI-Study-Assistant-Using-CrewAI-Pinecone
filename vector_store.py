from pinecone import Pinecone
from config import Config

pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(Config.PINECONE_INDEX_NAME)

def query_knowledge_base(query_embedding):
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
    # Join the text from metadata matches
    context = " ".join([item['metadata']['text'] for item in results['matches']])
    return context