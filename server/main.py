from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow all origins (Change this for security in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model (update path if necessary)
translator = pipeline("translation", model="../fine_tuned_model")


# Define Pydantic request model
class TranslationRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Igbo-to-English Translation API is running!"}


@app.post("/translate/")
def translate_text(request: TranslationRequest):
    translation = translator(request.text, max_length=100)
    return {"igbo": request.text, "english": translation[0]["translation_text"]}
