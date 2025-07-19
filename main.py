import os
import streamlit as st
import langchain
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
load_dotenv() 
st.title("News research Tool")
st.sidebar.title("News Article URLS")
urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL{i+1}")
    urls.append(url) 
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placefolder=st.empty()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=256,
    device=-1,
    framework="pt"  # force PyTorch
)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()
    text_splitter= RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',' '],
        chunk_size=1000
    )
    main_placefolder.text("Text Splitter... Started....")
    docs=text_splitter.split_documents(data)
    #Create embeddings
    vectorstore_openai = FAISS.from_documents(docs,embeddings)
    main_placefolder.text("Embedding vector started building....")
    time.sleep(2)
    with open(file_path,"wb") as f:
        pickle.dump(vectorstore_openai,f) 
query=main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore=pickle.load(f)
        chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever() )
        result=chain({"question":query},return_only_outputs=True)
        st.header("Answer")
        st.subheader(result["answer"])

