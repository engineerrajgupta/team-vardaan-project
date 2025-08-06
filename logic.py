import os
import json
import requests
import io
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- Initialize LLM and Embeddings ---
# We now use the Gemini API for both text generation and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)


# --- Core Logic Functions ---

def get_documents_from_pdf_url(pdf_url):
    """
    Downloads a PDF, extracts text page by page, and creates Document objects
    that store the text and the original page number in their metadata.
    """
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        documents = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                documents.append(Document(page_content=page_text, metadata={"source_page": i + 1}))
        print(f"PDF processed successfully. Found {len(documents)} pages with text.")
        return documents
    except Exception as e:
        print(f"Error processing PDF from URL: {e}")
        return None

def get_text_chunks(documents):
    """Splits Document objects into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def get_vector_store(text_chunks):
    """Creates a FAISS vector store from document chunks using Gemini embeddings."""
    if not text_chunks:
        print("Error: No text chunks to process.")
        return None
    try:
        print("Creating vector store with Gemini Embeddings...")
        # We pass the Gemini embedding model here instead of HuggingFace
        vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def llm_parser_extract_query_topic(user_question):
    """Uses the LLM to parse the user's question and extract the core topic."""
    prompt = f"""
    You are an expert at identifying the core subject of a question.
    Analyze the following user question and extract its main topic for semantic search.
    User question: "{user_question}"
    Return a JSON object with a single key "query_topic".
    Respond ONLY with the JSON object.
    """
    try:
        response = llm.invoke(prompt)
        json_string = response.content.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(json_string)
        if isinstance(parsed_json, list) and parsed_json:
            return parsed_json[0].get("query_topic", user_question)
        return parsed_json.get("query_topic", user_question)
    except Exception:
        return user_question

def generate_structured_answer(context_with_sources, question):
    """
    Generates a structured JSON answer including the answer, source quote, and page number.
    This is used for logging to prove the system's capabilities.
    """
    prompt = f"""
    You are a highly intelligent logic engine for analyzing legal and insurance documents.
    Your task is to answer the user's question based STRICTLY on the provided context.
    The context is a JSON object where keys are page numbers and values are the text from those pages.
    You must generate a structured JSON response.

    **Provided Context from Document:**
    ---
    {context_with_sources}
    ---

    **User's Question:**
    ---
    {question}
    ---

    **Your Task:**
    1. Find the single most relevant page and quote that answers the question.
    2. Generate a JSON object with the following schema:
    {{
      "question": "{question}",
      "answer": "A concise, direct answer to the question.",
      "source_quote": "The single, most relevant sentence from the context that directly supports your answer.",
      "source_page_number": "The page number (as an integer) where the source_quote was found."
    }}

    If the information is not in the context, respond with this JSON structure:
    {{
      "question": "{question}",
      "answer": "Information not found in the provided document context.",
      "source_quote": "N/A",
      "source_page_number": "N/A"
    }}
    """
    try:
        response = llm.invoke(prompt)
        json_string = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except Exception as e:
        print(f"An error occurred during LLM call: {e}")
        return {
            "question": question,
            "answer": "Failed to get a response from the language model.",
            "source_quote": f"API Error: {e}",
            "source_page_number": "N/A"
        }

def process_document_and_questions(pdf_url, questions):
    """
    Main processing pipeline. It now returns a simple list of answers to match
    the online judge's expected output format, while logging the detailed analysis.
    """
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents:
        return {"error": "Failed to retrieve or read the PDF document."}

    text_chunks = get_text_chunks(documents)
    vector_store = get_vector_store(text_chunks)
    if not vector_store:
        return {"error": "Failed to create the vector store."}

    final_simple_answers = []
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    print("\n--- DETAILED ANALYSIS LOG (FOR HUMAN REVIEW) ---")
    for i, question in enumerate(questions):
        print(f"\nProcessing question {i+1}/{len(questions)}: '{question}'")
        query_topic = llm_parser_extract_query_topic(question)
        retrieved_docs = retriever.get_relevant_documents(query_topic)
        
        context_with_sources = {}
        for doc in retrieved_docs:
            page = doc.metadata.get("source_page", "Unknown")
            if page not in context_with_sources:
                context_with_sources[page] = []
            context_with_sources[page].append(doc.page_content)
        
        context_json_str = json.dumps(context_with_sources, indent=2)

        if retrieved_docs:
            # Generate the detailed answer for logging purposes
            structured_answer = generate_structured_answer(context_json_str, question)
            print(json.dumps(structured_answer, indent=2))
            
            # Extract only the simple answer for the final response
            final_simple_answers.append(structured_answer.get("answer", "Error processing this question."))
        else:
            # Handle cases where no context is found
            error_answer = "Could not find any relevant context for this question in the document."
            print(f"  -> {error_answer}")
            final_simple_answers.append(error_answer)

    # This is the final object that will be sent to the judge
    final_response = {"answers": final_simple_answers}
    print("\n--- FINAL API RESPONSE (as sent to judge) ---")
    print(json.dumps(final_response, indent=2))
    return final_response
