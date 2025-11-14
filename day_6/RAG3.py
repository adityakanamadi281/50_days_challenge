# !pip install transformers faiss-cpu
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import faiss

# Load DPR encoder models
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder   = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Load tokenizers
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer  = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine Learning enables computers to learn from data.",
    "Natural Language Processing enables interaction with computers using human language.",
    "FAISS is used for similarity search and clustering of dense vectors."
]

query = "What is Natural Language Processing?"



document_embeddings = []

# Encode each document
for doc in documents:
    inputs = context_tokenizer(doc, return_tensors="pt")
    outputs = context_encoder(**inputs).pooler_output.detach().numpy()
    document_embeddings.append(outputs)

# Convert to matrix form (N, D)
document_embeddings = np.vstack(document_embeddings)

# Encode query
query_inputs = question_tokenizer(query, return_tensors="pt")
query_embedding = question_encoder(**query_inputs).pooler_output.detach().numpy()



embedding_dimension = document_embeddings.shape[1]

# L2 index (default FAISS index)
faiss_index = faiss.IndexFlatL2(embedding_dimension)

# Add document embeddings to FAISS
faiss_index.add(document_embeddings)



k = 3  # number of top results

distances, indices = faiss_index.search(query_embedding, k)


print("\n=== DPR + FAISS Results (Cosine/L2 Similarity) ===\n")

for idx, dist in zip(indices[0], distances[0]):
    print(f"Document: {documents[idx]}")
    print(f"Distance: {dist}\n")






# Reduce embeddings to 2D
pca = PCA(n_components=2)

# Fit PCA on document embeddings
reduced_embeddings = pca.fit_transform(document_embeddings)

# Transform query embedding
reduced_query_embedding = pca.transform(query_embedding)

# Plot the embeddings
plt.figure(figsize=(6, 6))

# Plot documents (blue)
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue')

# Plot query (red)
plt.scatter(reduced_query_embedding[:, 0], reduced_query_embedding[:, 1], color='red')

plt.title("Query vs Document Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.show()
