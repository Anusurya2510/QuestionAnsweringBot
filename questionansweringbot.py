# -*- coding: utf-8 -*-
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import StuffDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
import os
os.environ["OPENAI_API_KEY"] = "your openai-api-key"

# Load in our Text Data using TextLoader
loader = TextLoader("/geeta.txt")
# loader = PyPDFLoader("/random.pdf")
documents = loader.load()

# Chunk our data into smaller pieces for more effective vector search
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Example similarity search
docsearch.similarity_search("Geeta")

# Create the chain
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# Prompt the chain with a question
qa.run("Geeta was having affair with whom?")

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Iambic Pentameter:"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"])

# Incorporate the prompt into the chain
chain_type_kwargs = {"prompt": prompt}

qa_with_prompt = RetrievalQA.from_chain_type(llm=OpenAI(),
                                             chain_type="stuff",
                                             retriever=db.as_retriever(),
                                             chain_type_kwargs=chain_type_kwargs)

print(qa_with_prompt.run("Who is geeta?"))
