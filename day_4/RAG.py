# Vector Search


from langchain.schema import Document
from langchain.vectorstores import Chroma






movie_data = [
    {"title": "Inception", "description": "A mind-bending thriller about dream invasion.", "genre": "Sci-Fi"},
    {"title": "The Matrix", "description": "A hacker discovers the true nature of reality.", "genre": "Sci-Fi"},
    {"title": "Titanic", "description": "A tragic love story set aboard a doomed ocean linear", "genre": "Romance"},
    {"title": "The Godfather", "description": "The saga of a crime family in America.", "genre": "Crime"},
    {"title": "Intersteller", "description": "A space epic exploring love , survival and time.", "genre": "Sci-Fi"},
]


documents = [
    Document(
        page_content=movie["description"],
        metadata={"title": movie["title"], "genre": movie["genre"]}
    )
    for movie in movie_data
]



# Setup Embedding and ChromaDB
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(documents, embedding_model)



def vector_search(query, top_k=3):
    results = vector_store.similarity_search(query, k=top_k)
    return [
        {"title": res.metadata["title"], "description": res.page_content, "genre": res.metadata["genre"]}
        for res in results
    ]


def search_by_genre(genre, top_k=3):
    all_results = vector_store.similarity_search(query, k=10)
    filtered_results = [res for res in all_results if res.metadata["genre"]==genre]
    return filtered_results[:top_k]



# Semantic Search System for a Movie Database

# Query 1: General Vector Search
query1 = "A story about dreams and reality"
results1 = vector_search(query1, top_k=2)
print("General Vector Search Results:")
for movie in results1:
    print(f"Title: {movie['metadata']['title']}, Genre: {movie['genre']}, Description: {movie['description']}")

# Query 2: Filtered Search by Genre
query2 = "A thrilling space adventure"
genre_filter = "Sci-Fi"
results2 = search_by_genre(query2, genre=genre_filter, top_k=2)
print("\nFiltered Search by Genre Results:")
for movie in results2:
    print(f"Title: {movie['metadata']['title']}, Genre: {movie['genre']}, Description: {movie['description']}")


