# !pip install scikit-learn rank-bm25

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np


documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine learning enables computers to learn from data.",
    "Natural Language Processing enables interaction with computers using human language.",
    " FAISS is used for similarity search and clustering of dense vectors."
]


query = "What is Natural Language Processing?"

# Create vectorizer
vectorizer = TfidfVectorizer()

# Fit on documents and transform
doc_vectors = vectorizer.fit_transform(documents)

# Vector for the query
query_vector = vectorizer.transform([query])

# Compute cosine similarity
cosine_similarities = (doc_vectors @ query_vector.T).toarray().ravel()

# Get top 3 matches
top_k = 3
top_indices = np.argsort(cosine_similarities)[::-1][:top_k]

# Print results
print("\n=== TF-IDF Results ===\n")
for idx in top_indices:
    print(f"Document: {documents[idx]}")
    print(f"TF-IDF Score: {cosine_similarities[idx]}\n")



# Tokenize documents
tokenized_docs = [doc.split(" ") for doc in documents]

# Create BM25 object
bm25 = BM25Okapi(tokenized_docs)

# Tokenize query
tokenized_query = query.split(" ")

# Get BM25 scores
bm25_scores = bm25.get_scores(tokenized_query)

# Get top 3
top_indices = np.argsort(bm25_scores)[::-1][:top_k]

print("\n=== BM25 Results ===\n")
for idx in top_indices:
    print(f"Document: {documents[idx]}")
    print(f"BM25 Score: {bm25_scores[idx]}\n")
