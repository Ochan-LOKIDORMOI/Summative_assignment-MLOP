from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.prediction import load_model, predict, preprocess_image

# Initialize FastAPI application
app = FastAPI(
    title="Animal Classification API",
    description="API to classify uploaded images as Wild Animal or Domestic Animal.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],  # Flask app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
model_path = "models/pipe.pkl"
try:
    model = load_model(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(
        "Failed to load the model. Check the file path and format.")

@app.get("/")
def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"message": "API is running"}

@app.post("/predict")
async def predict_animal(file: UploadFile = File(...)):
    """
    Predict whether the uploaded image is a Wild Animal or Domestic Animal.
    """
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a JPEG or PNG image.",
            )

        # Read and preprocess the image
        contents = await file.read()
        image = preprocess_image(contents)

        # Perform prediction
        animal_type = predict(model, image)
        logger.info(f"Prediction: {animal_type} for file {file.filename}")

        return {"filename": file.filename, "animal_type": animal_type}

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
