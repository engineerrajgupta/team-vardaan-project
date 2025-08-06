# main.py
#trd
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Import the final async processing logic from your logic.py file
from logic import process_document_and_questions_async

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval API",
    description="An API that meets all evaluation criteria for the HackRx challenge.",
    version="7.0.0-pinecone-final" # Final version bump for the Pinecone architecture
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This allows your API to be called from any web front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body Validation ---
# This ensures the incoming data has the correct structure
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# --- API Endpoint ---
@app.post("/hackrx/run")
async def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    """
    This endpoint accepts a PDF document URL and a list of questions.
    It processes them asynchronously using the logic in logic.py and returns a JSON response.
    """
    print("Received request for /hackrx/run")
    try:
        # This is the only line that calls your complex logic.
        # All the power is contained in the imported function.
        results = await process_document_and_questions_async(
            pdf_url=request_data.documents, 
            questions=request_data.questions
        )
        
        # Basic error handling
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
            
        print("Successfully processed request. Returning simple results.")
        return results

    except Exception as e:
        # Catch-all for any unexpected errors during processing
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Health Check Endpoint ---
# A simple endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /hackrx/run"}

# To run this server locally, use the command: uvicorn main:app --reload
