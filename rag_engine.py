import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx2txt

# Config
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(os.path.dirname(__file__), "vector_db.pkl")

class RAGEngine:
    def __init__(self):
        print("Loading local embedding model (all-MiniLM-L6-v2)...")
        # Pure offline: This will download once and then use the cache
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.data = [] # List of {'text': ..., 'metadata': ...}
        self.embeddings = None
        self.load_db()

    def extract_text(self, file_path):
        """Extract text from different file types."""
        _, ext = os.path.splitext(file_path.lower())
        text = ""
        try:
            if ext == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif ext == ".pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
            elif ext == ".docx":
                text = docx2txt.process(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        return text

    def chunk_text(self, text, chunk_size=800, overlap=100):
        """Simple text chunking."""
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def index_data(self):
        """Scan DATA_DIR and index all documents."""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            return "Data directory created."

        files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
        if not files:
            return "No files found in data directory."

        all_chunks = []
        for file_name in files:
            file_path = os.path.join(DATA_DIR, file_name)
            text = self.extract_text(file_path)
            if not text: continue
            
            chunks = self.chunk_text(text)
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk,
                    'metadata': {'source': file_name}
                })
            print(f"Processed {file_name} ({len(chunks)} chunks)")

        if not all_chunks:
            return "No text could be extracted."

        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        texts = [c['text'] for c in all_chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.data = all_chunks
        
        self.save_db()
        return f"Successfully indexed {len(all_chunks)} chunks from {len(files)} documents."

    def save_db(self):
        with open(DB_PATH, 'wb') as f:
            pickle.dump({'data': self.data, 'embeddings': self.embeddings}, f)
        print("Database saved.")

    def load_db(self):
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'rb') as f:
                    db = pickle.load(f)
                    self.data = db['data']
                    self.embeddings = db['embeddings']
                print(f"Loaded {len(self.data)} chunks from database.")
            except Exception as e:
                print(f"Error loading database: {e}")

    def query(self, text, n_results=3):
        """Retrieve relevant context for a query using cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return ""

        query_embedding = self.model.encode([text])[0]
        
        # Calculate cosine similarities
        # similarities = np.dot(self.embeddings, query_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Manual cosine similarity for better stability
        norm_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        similarities = np.dot(norm_embeddings, norm_query)
        
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        
        context = ""
        for idx in top_indices:
            if similarities[idx] > 0.3: # Threshold
                item = self.data[idx]
                context += f"[Source: {item['metadata']['source']}]\n{item['text']}\n---\n"
        
        return context

if __name__ == "__main__":
    engine = RAGEngine()
    print(engine.index_data())
    print("\nTest Query: Who is Aman?")
    print(engine.query("Who is Aman?"))
