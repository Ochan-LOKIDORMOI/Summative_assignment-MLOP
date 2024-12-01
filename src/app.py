from flask import Flask, render_template, request
import requests
import base64
import logging

# Initialize Flask app
app = Flask(__name__, static_folder="static")
logging.basicConfig(level=logging.DEBUG)

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


if __name__ == "__main__":
    app.run(debug=True)
