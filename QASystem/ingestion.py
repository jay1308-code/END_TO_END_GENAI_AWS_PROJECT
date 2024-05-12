from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

import json
import os
import sys
import boto3## bedrock client

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)

def data_ingestion():
    loader = DirectoryLoader(r"C:\Users\Lenovo\Desktop\DESKTOP\END_TO_END_GENAI_AWS_PROJECT\data",glob="**/*.pdf",show_progress=True,loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    return docs

def get_vector_store(docs):
    vector_store_faiss= FAISS.from_documents(docs,bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss

if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)


   

