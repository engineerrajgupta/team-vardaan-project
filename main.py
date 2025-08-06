# main.py

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Import the new async processing logic
from Blogic import process_document_and_questions_async

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval API",
    description="An API that meets all evaluation criteria for the HackRx challenge.",
    version="4.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body Validation ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# --- API Endpoint ---
@app.post("/hackrx/run")
async def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    """
    This endpoint accepts a PDF document URL and a list of questions.
    It processes them asynchronously and returns a simple JSON response.
    """
    print("Received request for /hackrx/run")
    try:
        # Call the new asynchronous function
        results = await process_document_and_questions_async(
            pdf_url=request_data.documents, 
            questions=request_data.questions
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
            
        print("Successfully processed request. Returning simple results.")
        return results

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /hackrx/run"}

# To run this server, use the command: uvicorn main:app --reload
