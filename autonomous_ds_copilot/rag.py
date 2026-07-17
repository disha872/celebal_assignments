import os
import re
import math
from typing import List, Dict, Tuple

# Standard English stop words
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
    'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes',
    'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im',
    'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my',
    'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'shannt', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt',
    'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 'what',
    'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with',
    'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
}

def tokenize(text: str) -> List[str]:
    """Tokenizes text into a list of lowercase alphanumeric words, filtering out stop words."""
    # Convert to lowercase and match word tokens
    words = re.findall(r'[a-zA-Z0-9_]+', text.lower())
    return [w for w in words if w not in STOP_WORDS]

class DocChunk:
    """Represents a chunk of a documentation file."""
    def __init__(self, file_path: str, title: str, content: str):
        self.file_path = file_path
        self.title = title
        self.content = content
        self.tokens = tokenize(content + " " + title)
        self.tf: Dict[str, float] = {}  # Term frequencies

class RAGPipeline:
    """A pure-Python TF-IDF indexer and search retriever for RAG documentation."""
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.chunks: List[DocChunk] = []
        self.idf: Dict[str, float] = {}
        self.build_index()

    def build_index(self):
        """Loads docs, splits them into semantic chunks by headers, and calculates TF-IDF indices."""
        if not os.path.exists(self.docs_dir):
            return

        # 1. Load and chunk documents
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.md') or filename.endswith('.txt'):
                file_path = os.path.join(self.docs_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content by markdown headers (e.g., # or ##)
                sections = re.split(r'\n(?=#+ )', content)
                current_doc_title = filename
                
                for sec in sections:
                    sec = sec.strip()
                    if not sec:
                        continue
                    
                    # Extract header title as section title
                    header_match = re.match(r'^#+\s+(.+)$', sec.split('\n')[0])
                    section_title = header_match.group(1) if header_match else current_doc_title
                    
                    chunk = DocChunk(file_path, section_title, sec)
                    self.chunks.append(chunk)

        if not self.chunks:
            return

        # 2. Calculate Term Frequencies (TF) for each chunk
        for chunk in self.chunks:
            if not chunk.tokens:
                continue
            total_tokens = len(chunk.tokens)
            counts = {}
            for token in chunk.tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                chunk.tf[token] = count / total_tokens

        # 3. Calculate Inverse Document Frequencies (IDF)
        total_docs = len(self.chunks)
        all_vocab = set()
        for chunk in self.chunks:
            all_vocab.update(chunk.tf.keys())

        for token in all_vocab:
            # Number of documents containing token
            docs_with_token = sum(1 for chunk in self.chunks if token in chunk.tf)
            # Standard IDF formula
            self.idf[token] = math.log((1 + total_docs) / (1 + docs_with_token)) + 1

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[DocChunk, float]]:
        """Retrieves top_k relevant documentation chunks using Cosine Similarity over TF-IDF vectors."""
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        # 1. Calculate query TF-IDF
        query_counts = {}
        for token in query_tokens:
            query_counts[token] = query_counts.get(token, 0) + 1
        
        query_tf_idf = {}
        query_length_sq = 0.0
        for token, count in query_counts.items():
            tf = count / len(query_tokens)
            idf = self.idf.get(token, 0.0)
            val = tf * idf
            query_tf_idf[token] = val
            query_length_sq += val ** 2
        
        query_len = math.sqrt(query_length_sq)
        if query_len == 0:
            return []

        # 2. Calculate cosine similarity against all chunks
        results = []
        for chunk in self.chunks:
            dot_product = 0.0
            chunk_length_sq = 0.0
            
            # Compute pre-calculated chunk TF-IDF length and dot product
            # For efficiency, we iterate over the chunk vocabulary
            for token, tf in chunk.tf.items():
                idf = self.idf.get(token, 0.0)
                chunk_val = tf * idf
                chunk_length_sq += chunk_val ** 2
                if token in query_tf_idf:
                    dot_product += chunk_val * query_tf_idf[token]
            
            chunk_len = math.sqrt(chunk_length_sq)
            if chunk_len == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_len * chunk_len)
            
            results.append((chunk, similarity))

        # 3. Sort by similarity descending and filter out zero similarity
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return [r for r in results[:top_k] if r[1] > 0.0]

# Simple testing block
if __name__ == "__main__":
    # Test if it runs correctly
    rag = RAGPipeline("./docs")
    print(f"Total chunks indexed: {len(rag.chunks)}")
    test_query = "KeyError: 'Age' column mismatch"
    retrieved = rag.retrieve(test_query, top_k=2)
    for i, (chunk, score) in enumerate(retrieved):
        print(f"Rank {i+1} (Score: {score:.4f}) - File: {os.path.basename(chunk.file_path)} - Section: {chunk.title}")
        print("-" * 50)
        print(chunk.content[:200] + "...")
        print("=" * 50)
