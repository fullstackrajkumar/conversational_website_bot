from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


GROQ_API_KEY=os.getenv("GROQ_API_KEY")
MODEL_CODE=os.getenv("MODEL_CODE")

def extract_content(url):
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException:
        print("Failed to fetch url")
    soup = BeautifulSoup(response.text, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=' ')
    return ' '.join(text.split())

def splitting(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])

def ask_question(docs, question):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_CODE,
    )
    prompt = PromptTemplate.from_template(
        "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
    )
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context":docs, "question":question})
    return result



url = input("Enter the website URL: ")
raw_text = extract_content(url)
docs = splitting(raw_text)

while True:
    query = input("Ask your question about the page: ")
    print("\nQuestion : ",query)
    answer = ask_question(docs, query)
    print("\nAnswer:")
    print(answer)