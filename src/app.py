import os
import shutil
import zipfile
from flask import Flask, render_template, request
import requests
import base64
import logging
from src.model import retrain_model

# Initialize Flask app
app = Flask(__name__, static_folder="static")
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = 'uploaded_data'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000"


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Route for the homepage with upload functionality.
    Allows users to upload an image and sends it to the FastAPI server for prediction.
    """
    image_data = None
    prediction = None
    error = None

    if request.method == "POST":
        # Check if a file is uploaded
        if "file" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", error=error)

        file = request.files["file"]
        if file.filename == "":
            error = "No file selected."
            return render_template("index.html", error=error)

        try:
            # Read file data
            file_data = file.read()

            # Convert image to base64 for display
            image_data = base64.b64encode(file_data).decode("utf-8")

            # Send file to FastAPI server for prediction
            response = requests.post(
                f"{FASTAPI_URL}/predict",
                files={"file": (file.filename, file_data, file.content_type)},
            )

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("animal_type", "Unknown")
            else:
                # Handle errors from FastAPI
                error = response.json().get("detail", "Prediction failed")
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            error = f"Error: {str(e)}"

    # Render the template with results or errors
    return render_template(
        "index.html",
        image_data=image_data,
        prediction=prediction,
        error=error,
    )


@app.route("/about.html")
def about():
    return render_template("about.html")


@app.route("/graphs.html")
def graphs():
    return render_template("graphs.html")


@app.route("/retrain.html")
def retrain():
    return render_template("retrain.html")


@app.route('/upload', methods=['POST'])
def upload_folder():
    if 'file' not in request.files:
        return "No file part in the request.", 400

    file = request.files['file']

    # Check if the file is a zip archive
    if not file.filename.endswith('.zip'):
        return "Please upload a ZIP file containing the dataset.", 400

    # Save the uploaded ZIP file
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(zip_path)

    # Extract the ZIP file
    extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset')

    # Clear or create the extraction path
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)  # Remove the directory and its contents
    os.makedirs(extract_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Adjust for an extra folder layer (e.g., `retrain/` inside `uploaded_data/dataset`)
    subfolders = [d for d in os.listdir(
        extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    if len(subfolders) == 1:  # If a single folder exists, move its contents up
        top_level_folder = os.path.join(extract_path, subfolders[0])
        for item in os.listdir(top_level_folder):
            item_path = os.path.join(top_level_folder, item)
            shutil.move(item_path, extract_path)
        # Remove the now-empty top-level folder
        shutil.rmtree(top_level_folder)

    # Verify dataset structure
    categories = [d for d in os.listdir(
        extract_path) if os.path.isdir(os.path.join(extract_path, d))]
    if not categories:
        return "No valid categories (subdirectories) found in the dataset. Ensure the dataset is structured correctly.", 400

    # Log extracted directories
    print(f"Extracted categories: {categories}")

    # Check if categories contain valid images
    for category in categories:
        category_path = os.path.join(extract_path, category)
        if not os.listdir(category_path):
            return f"Category '{category}' is empty. Please add images to this folder.", 400

    # Retrain the model
    try:
        model = retrain_model(extract_path, categories)
    except Exception as e:
        return f"An error occurred during retraining: {str(e)}", 500

    return render_template('retrain.html', message="Model retrained successfully!")


if __name__ == "__main__":
    app.run(debug=True)
