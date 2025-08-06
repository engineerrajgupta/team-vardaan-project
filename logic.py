# logic.py

import os
import json
import requests
import io
import asyncio
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain.docstore.document import Document

# --- Load Environment Variables & Configuration ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# This MUST match the name of the index you created in the Pinecone dashboard
PINECONE_INDEX_NAME = "hackathon" 

# --- Initialize LLM and Embeddings ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=gemini_api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)


# --- Core Logic Functions ---

def get_documents_from_pdf_url(pdf_url):
    """Downloads and parses the PDF using the robust PyMuPDF library."""
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_doc = fitz.open(stream=response.content, filetype="pdf")
        documents = []
        for i, page in enumerate(pdf_doc):
            page_text = page.get_text()
            if page_text:
                documents.append(Document(page_content=page_text, metadata={"source_page": i + 1}))
        pdf_doc.close()
        print(f"PDF processed with PyMuPDF. Found {len(documents)} pages with text.")
        return documents
    except Exception as e:
        print(f"Error processing PDF with PyMuPDF from URL: {e}")
        return None

def get_text_chunks(documents):
    """Splits Document objects into smaller chunks using a recursive splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
    )
    return text_splitter.split_documents(documents)

async def get_vector_store_async(text_chunks):
    """
    Asynchronously connects to Pinecone and upserts the document chunks.
    This makes the vector store persistent in the cloud.
    """
    if not text_chunks: return None
    try:
        print(f"Connecting to Pinecone index '{PINECONE_INDEX_NAME}' and upserting documents...")
        vector_store = await Pinecone.afrom_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME
        )
        print("Pinecone upsert successful.")
        return vector_store
    except Exception as e:
        print(f"Error connecting to or upserting to Pinecone: {e}")
        return None

async def rerank_documents_with_llm_async(documents, question):
    """Re-ranks documents using a single, powerful LLM call."""
    if not documents: return ""
    docs_as_string = ""
    for i, doc in enumerate(documents):
        docs_as_string += f"--- DOCUMENT {i+1} ---\n{doc.page_content}\n\n"
    prompt = f"""
    You are a highly intelligent document analysis expert. Your task is to identify the most relevant document excerpts to answer a user's question.
    Below are several document excerpts, each marked with "--- DOCUMENT [number] ---".
    **User's Question:** "{question}"
    **Document Excerpts:** {docs_as_string}
    **Your Task:**
    Review all the document excerpts and identify the **top 3 most relevant excerpts** for answering the user's question.
    Return **ONLY the full, original text** of these top 3 excerpts, separated by the exact delimiter "---_---".
    Do not add any commentary, explanations, or numbering.
    """
    try:
        response = await llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"An error occurred during LLM re-ranking call: {e}")
        return ""

async def generate_simple_answer_async(context, question):
    """Generates a final answer from the clean, re-ranked context."""
    prompt = f"""
    You are a world-class logic engine. Your task is to answer the user's question based **STRICTLY AND ONLY** on the provided context.
    **Provided Context:** --- {context} ---
    **User's Question:** --- {question} ---
    **Your Task:**
    Generate a direct, concise, one-sentence answer.
    If the context **DOES NOT** contain the answer, respond with the exact phrase: "Information not found in the provided document context."
    """
    try:
        response = await llm.ainvoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"An error occurred during LLM generation call: {e}")
        return "Failed to get a response from the language model."

async def process_single_question_async(retriever, question):
    """Processes a single question using the efficient two-call LLM strategy."""
    print(f"Processing question: '{question}'")
    retrieved_docs = await retriever.aget_relevant_documents(question)
    if not retrieved_docs:
        return "Could not find any initial context for this question."
    print(f"  -> Re-ranking {len(retrieved_docs)} documents...")
    reranked_context = await rerank_documents_with_llm_async(retrieved_docs, question)
    if not reranked_context:
        return "Could not find relevant context after LLM re-ranking."
    print("  -> Generating final answer from re-ranked context...")
    answer = await generate_simple_answer_async(reranked_context, question)
    print(f"  -> Generated Answer: {answer}")
    return answer

async def process_document_and_questions_async(pdf_url, questions):
    """Main asynchronous processing pipeline for the entire request."""
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents: return {"error": "Failed to retrieve or read the PDF document."}
    text_chunks = get_text_chunks(documents)
    vector_store = await get_vector_store_async(text_chunks)
    if not vector_store: return {"error": "Failed to create the vector store."}
    base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    tasks = [process_single_question_async(base_retriever, q) for q in questions]
    final_simple_answers = await asyncio.gather(*tasks)
    final_response = {"answers": final_simple_answers}
    print("\n--- FINAL API RESPONSE (as sent to judge) ---")
    print(json.dumps(final_response, indent=2))
    return final_response
