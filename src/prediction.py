import io
import pickle
import numpy as np
from PIL import Image
from fastapi import HTTPException
import logging

# Set up logging for prediction
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """
    Load the model from a pickle file.

    Parameters:
        model_path (str): Path to the pickle file containing the model.

    Returns:
        The loaded model object.
    """
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess the input image data for prediction.

    Parameters:
        image_data (bytes): The raw image data.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    try:
        # Open the image and convert it to RGB
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Resize the image to the required size (e.g., 224x224)
        image = image.resize((150, 150))

        # Convert image to NumPy array and normalize
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]

        # Add a batch dimension (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Image preprocessing error")

def predict(model, image) -> str:
    """
    Predict the class of the input image using the model.

    Parameters:
        model: The loaded machine learning model.
        image (np.ndarray): Preprocessed image input.

    Returns:
        str: Predicted class (e.g., "Wild Animal", "Domestic Animal").
    """
    try:
        prediction = model.predict(image)
        logger.info(f"Model prediction: {prediction}")
        
        # Map prediction to class
        return "Wild Animal" if prediction[0][0] < 0.5 else "Domestic Animal"
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error")
