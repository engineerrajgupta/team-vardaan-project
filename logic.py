# logic.py (FINAL PERSISTENT VERSION)

import os
import json
import requests
import io
import asyncio
import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pinecone as pinecone_client
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
import redis # ADDED

# --- OCR Configuration ---
try:
    import pytesseract
    OCR_ENABLED = True
except ImportError:
    print("Warning: pytesseract not found. OCR capabilities will be disabled.")
    OCR_ENABLED = False

# --- Load Environment Variables & Configuration ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "hackathon" 
REDIS_URL = os.getenv("REDIS_URL") # ADDED

# --- Initialize LLM and Embeddings ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=gemini_api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)

# --- Initialize Redis Cache Connection ---
try:
    redis_client = redis.from_url(REDIS_URL)
    print("Successfully connected to Redis cache.")
except Exception as e:
    print(f"Warning: Could not connect to Redis. Caching will be disabled. Error: {e}")
    redis_client = None

# --- Document Processing Functions (Unchanged) ---
def get_documents_from_pdf_url(pdf_url):
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        documents = []
        total_text_length = 0
        for i, page in enumerate(pdf_doc):
            page_text = page.get_text()
            total_text_length += len(page_text)
            if page_text:
                documents.append(Document(page_content=page_text, metadata={"source_page": i + 1}))
        if OCR_ENABLED and total_text_length < 100 * len(pdf_doc):
            print("Low text detected. Attempting OCR fallback...")
            documents = []
            for i, page in enumerate(pdf_doc):
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text:
                    documents.append(Document(page_content=ocr_text, metadata={"source_page": i + 1, "ocr": True}))
            print(f"OCR processed {len(documents)} pages.")
        pdf_doc.close()
        print(f"PDF processed. Found {len(documents)} pages with text.")
        return documents
    except Exception as e:
        print(f"Error processing PDF from URL: {e}")
        return None

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# --- Main Processing Pipeline ---
async def process_single_question_fast(retriever, question):
    print(f"  -> Processing fast: '{question}'")
    retrieved_docs = await retriever.ainvoke(question)
    if not retrieved_docs:
        return "Information not found in the provided document context."
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    You are a logic engine. Answer the user's question based **STRICTLY AND ONLY** on the provided context.
    **Provided Context:** --- {context} ---
    **User's Question:** --- {question} ---
    **Your Task:** Generate a direct, concise answer. If the context does not contain the answer, respond with the exact phrase: "Information not found in the provided document context."
    """
    response = await llm.ainvoke(prompt)
    return response.content.strip()

async def process_document_and_questions_async(pdf_url, questions):
    """Main entry point for the API, now with persistent Redis caching."""
    
    # --- THE FIX IS HERE: Use Redis for persistent caching ---
    is_processed = redis_client.exists(pdf_url) if redis_client else False
    
    if not is_processed:
        print(f"--- New URL detected. Starting full ingestion for {pdf_url} ---")
        documents = get_documents_from_pdf_url(pdf_url)
        if not documents: return {"error": "Failed to retrieve or read the PDF document."}
        
        text_chunks = get_text_chunks(documents)
        
        print(f"--- Embedding and upserting {len(text_chunks)} chunks to Pinecone. This may take time... ---")
        await PineconeVectorStore.afrom_documents(
            documents=text_chunks, embedding=embeddings, index_name=PINECONE_INDEX_NAME
        )
        
        # Add the URL to the persistent Redis cache with an expiration time (e.g., 24 hours)
        if redis_client:
            redis_client.set(pdf_url, "processed", ex=86400)
        print("--- Full ingestion complete. Subsequent requests for this URL will be fast. ---")
    else:
        print(f"--- URL found in Redis cache. Skipping ingestion for {pdf_url} ---")

    pc = pinecone_client.Pinecone(api_key=pinecone_api_key)
    index = pc.Index(PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    tasks = [process_single_question_fast(retriever, q) for q in questions]
    final_answers = await asyncio.gather(*tasks)
    
    final_response = {"answers": final_answers}
    print("\n--- FINAL API RESPONSE (as sent to judge) ---")
    print(json.dumps(final_response, indent=2))
    return final_response
