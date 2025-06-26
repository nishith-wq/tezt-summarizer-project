import re
import numpy as np
import networkx as nx
from collections import defaultdict
from math import log
from heapq import nlargest

# Custom tokenization functions
def sentence_tokenize(text):
    """Split text into sentences using regex"""
    return [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text) if s.strip()]

def word_tokenize(sentence):
    """Tokenize sentence into words"""
    return re.findall(r'\b\w+\b', sentence.lower())

def get_stopwords():
    """Return a predefined set of common English stopwords"""
    return {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
        'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
        'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
        'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
        'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
        "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }

def build_tfidf_matrix(sentences):
    """Build TF-IDF matrix from sentences"""
    stopwords = get_stopwords()
    word_freq = defaultdict(int)
    doc_freq = defaultdict(int)
    
    # Preprocess all sentences and count frequencies
    preprocessed = []
    for sent in sentences:
        words = [word for word in word_tokenize(sent) if word not in stopwords]
        preprocessed.append(words)
        for word in set(words):
            doc_freq[word] += 1
    
    # Calculate total word frequency
    for words in preprocessed:
        for word in words:
            word_freq[word] += 1
    
    # Create TF-IDF vectors
    vectors = []
    total_docs = len(sentences)
    
    for words in preprocessed:
        vector = {}
        word_count = len(words)
        for word in words:
            # TF calculation
            tf = word_freq[word] / word_count
            # IDF calculation
            idf = log(total_docs / (1 + doc_freq[word]))
            vector[word] = tf * idf
        vectors.append(vector)
    
    return vectors

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1) | set(vec2))
    norm1 = sum(val ** 2 for val in vec1.values()) ** 0.5
    norm2 = sum(val ** 2 for val in vec2.values()) ** 0.5
    
    if norm1 * norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def text_summarizer(text, num_sentences=3):
    """
    Summarize text using TextRank algorithm without external NLP libraries
    
    Args:
        text (str): Input text to summarize
        num_sentences (int): Number of sentences in summary
        
    Returns:
        str: Generated summary
    """
    if not text or not text.strip():
        return "Error: Input text is empty"
    
    # Tokenize sentences
    sentences = sentence_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    
    # Build TF-IDF vectors
    vectors = build_tfidf_matrix(sentences)
    
    # Create similarity matrix
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = cosine_similarity(vectors[i], vectors[j])
    
    # Apply PageRank algorithm
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph, max_iter=1000)
    
    # Get top ranked sentences
    ranked_indices = nlargest(num_sentences, range(len(scores)), key=lambda i: scores[i])
    ranked_indices.sort()  # Maintain original order
    
    # Generate summary
    return " ".join(sentences[i] for i in ranked_indices)

# Example usage
if __name__ == "__main__":
    sample_text = """
    Natural language processing (NLP) is a subfield of artificial intelligence that focuses 
    on enabling computers to understand human language. Modern NLP systems use deep learning 
    models like transformers to analyze text. Key applications include translation, sentiment 
    analysis, and chatbots. These models have achieved impressive results but still face 
    challenges with contextual understanding and rare languages.
    """
    
    print("Summary:")
    print(text_summarizer(sample_text, num_sentences=2))