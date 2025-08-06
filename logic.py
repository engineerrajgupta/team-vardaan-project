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
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Load Environment Variables ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# --- Initialize LLM and Embeddings ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=gemini_api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)


# --- Core Logic Functions (Unchanged and Still Optimal) ---

def get_documents_from_pdf_url(pdf_url):
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
    )
    return text_splitter.split_documents(documents)

async def get_vector_store_async(text_chunks):
    if not text_chunks: return None
    try:
        print("Creating vector store...")
        vector_store = await FAISS.afrom_documents(documents=text_chunks, embedding=embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# --- NEW: Single-Call LLM Re-ranking Function ---
async def rerank_documents_with_llm_async(documents, question):
    """
    Re-ranks a list of documents based on their relevance to a question using a single LLM call.
    """
    if not documents:
        return ""

    # Create a numbered list of documents for the prompt
    docs_as_string = ""
    for i, doc in enumerate(documents):
        docs_as_string += f"--- DOCUMENT {i+1} ---\n{doc.page_content}\n\n"

    # This prompt asks the LLM to act as a relevance filter
    prompt = f"""
    You are a highly intelligent document analysis expert. Your task is to identify the most relevant document excerpts to answer a user's question.
    Below are several document excerpts, each marked with "--- DOCUMENT [number] ---".

    **User's Question:**
    "{question}"

    **Document Excerpts:**
    {docs_as_string}

    **Your Task:**
    Review all the document excerpts and identify the **top 3 most relevant excerpts** for answering the user's question.
    Return **ONLY the full, original text** of these top 3 excerpts, separated by the exact delimiter "---_---".
    Do not add any commentary, explanations, or numbering.
    """
    
    try:
        response = await llm.ainvoke(prompt)
        # The returned content will be the top 3 docs separated by our delimiter
        return response.content.strip()
    except Exception as e:
        print(f"An error occurred during LLM re-ranking call: {e}")
        return ""

# --- Final Answer Generation Function (Unchanged) ---
async def generate_simple_answer_async(context, question):
    prompt = f"""
    You are a world-class logic engine. Your task is to answer the user's question based **STRICTLY AND ONLY** on the provided context.

    **Provided Context:**
    ---
    {context}
    ---

    **User's Question:**
    ---
    {question}
    ---

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

# --- UPDATED: Main Processing Logic for a Single Question ---
async def process_single_question_async(retriever, question):
    """
    Processes a single question using the efficient two-call LLM strategy.
    """
    print(f"Processing question: '{question}'")
    
    # 1. Fast initial retrieval
    retrieved_docs = await retriever.aget_relevant_documents(question)
    
    if not retrieved_docs:
        error_answer = "Could not find any initial context for this question."
        print(f"  -> {error_answer}")
        return error_answer

    # 2. LLM Call 1: Re-rank and filter the documents
    print(f"  -> Re-ranking {len(retrieved_docs)} documents...")
    reranked_context = await rerank_documents_with_llm_async(retrieved_docs, question)
    
    if not reranked_context:
        error_answer = "Could not find relevant context after LLM re-ranking."
        print(f"  -> {error_answer}")
        return error_answer

    # 3. LLM Call 2: Generate the final answer from the high-quality context
    print("  -> Generating final answer from re-ranked context...")
    answer = await generate_simple_answer_async(reranked_context, question)
    print(f"  -> Generated Answer: {answer}")
    return answer

# --- Main Pipeline Orchestrator (Updated to use the new retriever) ---
async def process_document_and_questions_async(pdf_url, questions):
    """
    Main asynchronous processing pipeline with the FAST and ACCURATE re-ranking strategy.
    """
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents: return {"error": "Failed to retrieve or read the PDF document."}

    text_chunks = get_text_chunks(documents)
    vector_store = await get_vector_store_async(text_chunks)
    if not vector_store: return {"error": "Failed to create the vector store."}

    # This retriever will fetch a larger number of docs for the re-ranking step.
    base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    tasks = [process_single_question_async(base_retriever, q) for q in questions]
    final_simple_answers = await asyncio.gather(*tasks)

    final_response = {"answers": final_simple_answers}
    print("\n--- FINAL API RESPONSE (as sent to judge) ---")
    print(json.dumps(final_response, indent=2))
    return final_response
