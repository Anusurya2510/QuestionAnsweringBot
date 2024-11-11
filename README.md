# Question Answering Bot

A sophisticated question-answering system built with LangChain that combines document retrieval, embeddings, and language models to provide contextual answers to questions.

## Features

- Document loading and processing with TextLoader
- Text chunking for optimal vector search
- Similarity-based document retrieval using Chroma vector store
- Integration with OpenAI's language models
- Custom prompt templating support
- HuggingFace embeddings integration
- Configurable retrieval parameters

## Prerequisites

- Python 3.x
- OpenAI API key

## Required Dependencies

```bash
pip install langchain chromadb openai transformers torch
