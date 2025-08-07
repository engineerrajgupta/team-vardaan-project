# logic.py (FINAL UPGRADED VERSION)

import os
import json
import requests
import io
import fitz  # UPGRADED: Using PyMuPDF for better text and table extraction
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# UPGRADED: Using a smarter text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- Initialize LLM and Embeddings ---
# UPGRADED: Using the more powerful Pro model for better reasoning in the final answer
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0)
# UPGRADED: Using the latest embedding model for higher quality retrieval
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)


# --- Core Logic Functions ---

# UPGRADED: Switched to PyMuPDF for superior parsing of text and tables
def get_documents_from_pdf_url(pdf_url):
    """
    Downloads a PDF, extracts text page by page using the robust PyMuPDF library,
    and creates Document objects with page number metadata.
    """
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

# UPGRADED: Using a smarter splitter to keep sentences and paragraphs together
def get_text_chunks(documents):
    """Splits Document objects into smaller, semantically meaningful chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def generate_simple_answer(context, question):
    """
    Generates a direct, concise answer based on the provided context.
    This is the only LLM call per question, optimized for speed.
    """
    prompt = f"""
    You are a highly intelligent logic engine. Your task is to answer the user's question based **STRICTLY AND ONLY** on the provided context.

    **Provided Context from Document:**
    ---
    {context}
    ---

    **User's Question:**
    ---
    {question}
    ---

    **Your Task:**
    Generate a direct, concise, one-sentence answer to the user's question.
    If the context **DOES NOT** contain the answer, you **MUST** respond with the exact phrase: "Information not found in the provided document context."
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        return "Failed to get a response from the language model."

def process_document_and_questions(pdf_url, questions):
    """
    Main processing pipeline. It now returns a simple list of answers to match
    the online judge's expected output format.
    """
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents:
        # If PDF processing fails (e.g., scanned PDF), return "not found" for all questions
        return {"answers": ["Information not found in the provided document context."] * len(questions)}

    text_chunks = get_text_chunks(documents)
    if not text_chunks:
        return {"answers": ["Failed to chunk the document text."] * len(questions)}

    # Create embeddings for all text chunks in memory
    chunk_texts = [chunk.page_content for chunk in text_chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    final_simple_answers = []

    for question in questions:
        # Embed the current question
        query_embedding = embeddings.embed_query(question)

        # --- Manual Similarity Search using NumPy (The Fast Core Logic) ---
        similarities = [np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)) for chunk_emb in chunk_embeddings]
        
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_docs = [text_chunks[i] for i in top_indices]
        
        # Combine the content of the top chunks into a single context string
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        if retrieved_docs:
            answer = generate_simple_answer(context, question)
            final_simple_answers.append(answer)
        else:
            final_simple_answers.append("Could not find any relevant context for this question in the document.")

    final_response = {"answers": final_simple_answers}
    print("\n--- FINAL API RESPONSE (as sent to judge) ---")
    print(json.dumps(final_response, indent=2))
    return final_response
