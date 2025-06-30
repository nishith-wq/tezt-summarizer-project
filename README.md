# tezt-summarizer-project

*Company* - *CODTECH IT SOLUTIONS*

*NAME* - *NISHEETH CHANDRA*

*INTERN ID* - *CT04DF2787*

*DOMAIN* - *ARTIFICIAL INTELLIGENCE*

*DURATION* - *4 WEEKS*

*MENTOR* - *NEELA SANTHOSH*
## Text Summarization using Custom TextRank in Python

This project implements **extractive text summarization** using a custom-built version of the **TextRank algorithm**, entirely without external NLP libraries like spaCy or NLTK. It uses TF-IDF and cosine similarity to build a sentence similarity graph and applies PageRank to extract the most important sentences.

---

## Features

- Custom sentence and word tokenization using regular expressions  
- Built-in stopword removal  
- TF-IDF based sentence vectorization  
- Cosine similarity matrix between sentences  
- Graph-based ranking using PageRank  
- Clean extractive summary output  
- No external NLP dependencies  

---

## How It Works

1. **Sentence Tokenization** - 
   Breaks the input text into sentences using a smart regex.

2. **Word Tokenization**  
   Converts each sentence to lowercase words, removing stopwords and punctuation.

3. **TF-IDF Matrix Construction**
   - **TF (Term Frequency)**: Measures word frequency in a sentence.  
   - **IDF (Inverse Document Frequency)**: Measures word uniqueness across all sentences.  
   - Each sentence becomes a TF-IDF vector.

4. **Cosine Similarity Matrix**  
   Builds an `n x n` matrix where each cell holds the cosine similarity between two sentence vectors.

5. **PageRank Algorithm**  
   Constructs a graph where each node is a sentence. Edges are weighted by cosine similarity. PageRank determines the most important sentences.

6. **Summary Extraction**  
   Selects top-ranked sentences and joins them in the order of appearance to produce a final summary.

---

## Tech Stack

- Python standard libraries: `re`, `collections`, `math`, `heapq`
- `numpy` for similarity matrix
- `networkx` for PageRank implementation

---

## Use Cases

- Quick summarization of research papers or articles  
- Content preview generation for long documents  
- NLP learning tools for students and beginners  
- Lightweight summarization for embedded/edge applications

---

## Why This is Cool

- No fancy NLP libraries – everything built from scratch  
- Great educational value for understanding NLP fundamentals  
- Lightweight and portable for small systems  
- Easily extensible for multi-document summarization

---

## Example Usage

```python
from summarizer import text_summarizer

text = """Natural language processing (NLP) is a subfield of artificial intelligence..."""
summary = text_summarizer(text, num_sentences=3)
print(summary)
