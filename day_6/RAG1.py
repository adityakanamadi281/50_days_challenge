# Vector - Based Retrieval 
# # Install required packages
# !pip install langchain faiss-cpu sentence-transformers huggingface-hub langchain-community


import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS



documents = [
    Document(page_content="Artificial Intelligence is transforming industries."),
    Document(page_content="Machine learning enables computers to learn from data."),
    Document(page_content="Natural language processing helps computers understand text."),
    Document(page_content="FAISS is used for efficient similarity search and clustering of dense vectors.")
]


# You can replace the model with any other sentence transformer model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

texts = [doc.page_content for doc in documents]

# Convert documents to embeddings
doc_embeddings = np.array([embedding_model.embed_query(text) for text in texts])

# Normalise for cosine similarity
doc_embeddings = np.array([emb / np.linalg.norm(emb) for emb in doc_embeddings])



dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(doc_embeddings)

query = "Explain the role of FAISS in vector retrieval."

query_embedding = embedding_model.embed_query(query)
query_embedding = query_embedding / np.linalg.norm(query_embedding)


k = 3  # number of results
distances, indices = index.search(np.array([query_embedding]), k)


print(f"Query: {query}\n")
print("Top Matches:\n")

for i in range(k):
    print(f"Document: {documents[indices[0][i]].page_content}")
    print(f"Distance: {distances[0][i]}")
    print()

