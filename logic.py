import os
import json
import requests
import io
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

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
                # The 'source' metadata will help us track the page number
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
    """Creates a FAISS vector store from document chunks."""
    if not text_chunks:
        print("Error: No text chunks to process.")
        return None
    try:
        print("Creating vector store...")
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.from_documents(documents=text_chunks, embedding=embedding)
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
    Generates a structured JSON answer. It uses the context if available, or its
    own intelligence if the context is empty, clearly stating the situation.
    """
    prompt = f"""
    You are a highly intelligent logic engine for analyzing legal and insurance documents.
    Your task is to answer the user's question based on the provided context.

    **Provided Context from Document:**
    ---
    {context_with_sources}
    ---
    (Note: The context above may be empty if no relevant text was found in the document.)

    **User's Question:**
    ---
    {question}
    ---

    **Your Task:**
    1.  Answer the question using the provided context.
    2.  If the context is insufficient, use your general knowledge of insurance policies to provide the most likely answer.
    3.  Generate a JSON object with the following schema:
    {{
      "question": "{question}",
      "answer": "A concise, direct answer to the question. Do not mention the provided context or document in your answer.",
      "source_quote": "The single, most relevant sentence from the context that directly supports your answer. If no direct quote is available, put 'N/A'.",
      "source_page_number": "The page number (as an integer) where the source_quote was found. If not applicable, put 'N/A'."
    }}

    Example of a good inferential answer when context is empty:
    {{
        "question": "What is the policy on alien abductions?",
        "answer": "Standard insurance policies typically do not cover events like alien abductions unless explicitly stated in a special addendum.",
        "source_quote": "N/A",
        "source_page_number": "N/A"
    }}

    Now, generate the JSON response.
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
    Main processing pipeline. It now always calls the generation step to provide
    more intelligent answers, even if the initial search finds no direct context.
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

        # Always generate an answer, even if context is empty.
        # The intelligence is now in the prompt of the generation function.
        structured_answer = generate_structured_answer(context_json_str, question)
        print(json.dumps(structured_answer, indent=2))
        
        # Extract only the simple answer for the final response
        final_simple_answers.append(structured_answer.get("answer", "Error processing this question."))

    # This is the final object that will be sent to the judge
    final_response = {"answers": final_simple_answers}
    print("\n--- FINAL API RESPONSE (as sent to judge) ---")
    print(json.dumps(final_response, indent=2))
    return final_response
