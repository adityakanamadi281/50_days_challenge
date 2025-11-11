# Image Embeddings

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import torch

feature_extractor= ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

image= Image.open(r"C:\Users\adity\50_days_challenge\day_3\images\ml_eda_histograms.png").convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    image_embedding = outputs.last_hidden_state.mean(dim=1)
print(image_embedding.shape)
print(image_embedding)






# Audio Embeddings

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import librosa

# from moviepy.editor import VideoFileClip
# input_path = r"C:\Users\adity\50_days_challenge\day_3\audio\ಕಾಂತಾರ 1 ｜ Gottilla Shivane ｜ Brahmakalasha ｜ Divinesong ｜ Kantara..webm"
# output_path = r"C:\Users\adity\50_days_challenge\day_3\audio\ಕಾಂತಾರ 1.wav"

# video = VideoFileClip(input_path)
# video.audio.write_audiofile(output_path, codec="pcm_s16le")
# video.close()

feature_extractor_1 = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model1 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

audio_file = r"C:\Users\adity\50_days_challenge\day_3\audio\ಕಾಂತಾರ 1.wav"
audio, rate = librosa.load(audio_file, sr=16000)
inputs1 = feature_extractor_1(audio, return_tensors="pt", sampling_rate=rate)

with torch.no_grad():
    outputs1 = model1(**inputs1)
    audio_embedding = outputs1.last_hidden_state.mean(dim=1)
print(audio_embedding.shape)
print(audio_embedding)








# Text Embeddings 
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model2 = AutoModel.from_pretrained("bert-base-uncased")

text = "This is an example for sentence embedding."
inputs2 = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs2 = model2(**inputs2)
    text_embedding = outputs2.last_hidden_state.mean(dim=1)
print(text_embedding.shape)
print(text_embedding)




# Combining Embeddings
# combined_embedding = torch.cat([image_embedding, audio_embedding, text_embedding])
# print("Combined_embedding shape : ", combined_embedding.shape)







# Storing Embeddings in Vector Databases
#ChromaDB

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

chroma_db = Chroma(persist_directory="./chroma_db",
                   embedding_function=embedding_model)

documents = [
    Document(page_content="Elon Musk Leads Tesla and SpaceX", metadata={"source": "1"}),
    Document(page_content="Tesla's mission is to accelerate the world's transition to sustainable energy.", metadata={"source": "2"}),
    Document(page_content="SpaceX aims to make space travel more affordable and accessible.", metadata={"source": "3"}),
]

chroma_db.add_documents(documents)

query_text = "What is Tesla's mission?"
results = chroma_db.similarity_search(query_text, k=2)

for idx, result in enumerate(results, 1):
    print(f"result{idx}: {result.page_content}")
    print(f"Content : {result.page_content}")
    print(f"Metadata : {result.metadata}")




# FAISS 
# pip install faiss-cpu

import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {"id": "1", "text": "Elon Musk owner of Spacex and tesla."},
    {"id": "2", "text": "Tesla's mission is to accelerate the worlds transition to sustainable energy."},
    {"id": "3", "text": "SpaceX aims to make space travel more affordable and accessible"},
]

doc_texts = [doc["text"] for doc in documents]
doc_ids = [doc["id"] for doc in documents]
embeddings = embedding_model.encode(doc_texts)


embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

index.add(embeddings)

query_text = "What is Tesla's mission?"
query_embedding = embedding_model.encode([query_text])

k=2
distances, indices = index.search(query_embedding, k)

print("Query:", query_text)
for i, idx in enumerate(indices[0]):
    print(f"\nResult{i+1}:")
    print(f"Text: {doc_texts[idx]["text"]}")
    print(f"ID: {documents[idx]["id"]}")
    print(f"Distance: {distances[0][i]:.4f}")
    