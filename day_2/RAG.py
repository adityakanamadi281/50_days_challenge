# pip install langchain sentence-transformers chromadb

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import retrieval_qa
from langchain_core.prompts import ChatPromptTemplate



import os
from getpass import getpass
HF_token = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HF_token


URL = "https://www.langchain.com/"

data =WebBaseLoader(URL)
content = data.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,chunk_overlap=0)
chunking = text_splitter.split_documents(content)

len(chunking)


embeddings= HuggingFaceInferenceAPIEmbeddings(
    api_key = HF_token,model_name="BAAI/bge-en-v1.5"
)

vectorstore= Chroma.from_documents(chunking,embeddings)

llm = HuggingFaceHub(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    model_kwargs = {"max_new_tokens":1024,
                   "temperature":0.1,
                   "repitition_penalty":1.1,
                   "return_full_text":False}
)

response = llm.invoke("who is mathematician?")



template = ChatPromptTemplate.from_messages(
    [
        (
            "system","You are an math assistant,you only answer questions and nothing else"
        ),
        (
            "user", "{query}"
        )
    ]
)


prompt = template.format_messages(query="what is square root of 555?")
response = llm.invoke(prompt)


template2 = ChatPromptTemplate.from_template(""" <|system|> You are an math assistant,you only answer questions and nothing else <|user|> {query} <|assistant|> """)



