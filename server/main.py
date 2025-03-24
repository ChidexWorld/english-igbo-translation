from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

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
