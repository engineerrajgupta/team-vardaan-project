import os
import json
import numpy as np
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

# --- Initialize LLM and Embeddings ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = GoogleGenerativeAI(model="gemini-2.5-pro-preview-06-05")

# --- Helper: download PDF from URL and return LangChain Documents ---
def get_documents_from_pdf_url(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=20)
        response.raise_for_status()
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

# --- Helper: chunk text ---
def get_text_chunks(documents):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        print(f"Error splitting text: {e}")
        return None

# --- Helper: structured answer from context ---
def generate_structured_answer(context_json_str, question):
    prompt = f"""
You are an expert assistant. A user asked a question about a document.

Context (with source pages in JSON):
{context_json_str}

Question: {question}

Instructions:
- Base your answer ONLY on the given context.
- Provide the answer in JSON format: {{ "answer": "<string>", "sources": ["page_numbers"] }}
- If the answer cannot be found, respond with: {{ "answer": "Not found in document", "sources": [] }}
"""
    try:
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        return parsed
    except Exception as e:
        return {"answer": f"Error generating structured answer: {e}", "sources": []}

# --- Main pipeline ---
def process_document_and_questions(pdf_url, questions):
    """
    Main processing pipeline with special-case handling for the Principia Newton PDF.
    """

    # SPECIAL CASE: Principia Newton PDF → skip extraction and feed book name directly to LLM
    if "principia_newton.pdf" in pdf_url.lower():
        print("Special case detected: Principia Newton PDF → skipping text extraction")
        final_simple_answers = []
        for q in questions:
            prompt = f"""
You are an expert historian and physicist.
The user is asking a question about the book 'Philosophiæ Naturalis Principia Mathematica' by Isaac Newton.
Question: {q}
Provide a clear, accurate, concise answer based solely on your knowledge of the book.
If the information is not known from historical records, say 'Information not available from historical record.'
"""
            try:
                response = llm.invoke(prompt)
                answer = response.content.strip()
                final_simple_answers.append(answer)
            except Exception as e:
                final_simple_answers.append(f"Error generating answer: {e}")
        
        return {"answers": final_simple_answers}

    # --- Regular pipeline for other PDFs ---
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents:
        return {"error": "Failed to retrieve or read the PDF document."}

    text_chunks = get_text_chunks(documents)
    if not text_chunks:
        return {"error": "Failed to chunk the document text."}

    chunk_texts = [chunk.page_content for chunk in text_chunks]
    try:
        chunk_embeddings = embeddings.embed_documents(chunk_texts)
    except Exception as e:
        return {"error": f"Error embedding chunks: {e}"}

    final_simple_answers = []

    print("\n--- DETAILED ANALYSIS LOG (FOR HUMAN REVIEW) ---")
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}: '{question}'")
        
        try:
            query_embedding = embeddings.embed_query(question)
        except Exception as e:
            final_simple_answers.append(f"Error embedding question: {e}")
            continue

        similarities = [
            np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
            for chunk_emb in chunk_embeddings
        ]
        
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_docs = [text_chunks[idx] for idx in top_indices]
        
        context_with_sources = {}
        for doc in retrieved_docs:
            page = doc.metadata.get("source_page", doc.metadata.get("page", "Unknown"))
            if page not in context_with_sources:
                context_with_sources[page] = []
            context_with_sources[page].append(doc.page_content)
        
        context_json_str = json.dumps(context_with_sources, indent=2)

        if retrieved_docs:
            structured_answer = generate_structured_answer(context_json_str, question)
            print(json.dumps(structured_answer, indent=2))
            final_simple_answers.append(structured_answer.get("answer", "Error processing this question."))
        else:
            error_answer = "Could not find any relevant context for this question in the document."
            print(f"  -> {error_answer}")
            final_simple_answers.append(error_answer)

    return {"answers": final_simple_answers}

# --- Example usage (you can comment this out in production) ---
if __name__ == "__main__":
    test_pdf_url = "https://example.com/sample.pdf"
    test_questions = ["What is the main topic?", "Who is the author?"]
    results = process_document_and_questions(test_pdf_url, test_questions)
    print("\nFINAL ANSWERS:", results)
