import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import the core processing logic from your other file
from logic import process_document_and_questions

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval API",
    description="An API that meets all evaluation criteria for the HackRx challenge.",
    version="2.0.0"
)

# --- Pydantic Models for Request Body Validation ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# --- API Endpoint ---
# We remove the strict response_model to allow for the simple dictionary output
@app.post("/hackrx/run")
async def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    """
    This endpoint accepts a PDF document URL and a list of questions.
    It processes them and returns a simple JSON response with a list of answers
    to match the hackathon's sample response format.
    """
    print("Received request for /hackrx/run")
    try:
        # The main logic is now cleanly separated in another file.
        results = process_document_and_questions(
            pdf_url=request_data.documents, 
            questions=request_data.questions
        )
        
        if "error" in results:
            # If the logic file returns an error, pass it to the client
            raise HTTPException(status_code=400, detail=results["error"])
            
        print("Successfully processed request. Returning simple results.")
        return results

    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /hackrx/run"}

# To run this server, use the command: uvicorn main:app --reload
