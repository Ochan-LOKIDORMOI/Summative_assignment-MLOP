# Animal Classification System

## Overview
This project provides a system to classify animals as **domestic** or **wild** using image-based predictions. It features an end-to-end solution for model inference, retraining, and visualization. 

### Key Features
- **Image Classification**: Predicts if an uploaded image depicts a domestic or wild animal.
- **Model Training**: Enables users to train or retrain models using custom datasets.
- **Interactive Visualizations**: Displays model performance metrics such as training loss, accuracy, and confusion matrices.
- **REST API**: Provides endpoint-based interaction for seamless integration with external services.
- **Scalable Deployment**: Dockerized infrastructure for cloud deployment, tested locally.

---

## File Structure

```plaintext
animal-classification/
├── README.md          # Documentation
├── notebook/          # Jupyter Notebooks for initial experiments
│   └── initial_model_training.ipynb
├── src/               # Source code for the project
│   ├── preprocessing.py  # Image preprocessing utilities
│   ├── model.py          # ML model definition and training logic
│   └── prediction.py     # Model inference and prediction
├── data/              # Dataset directory
│   ├── wild/           
│   └── domestic/      
├── models/            # Trained model storage
│   ├── current_model.pkl
│   └── backup_model.pkl
```

# **Installation Instructions**
## Prerequisites
- Python 3.8 or higher
- Virtual environment setup (venv or conda)
- Docker (optional for deployment)

# Step 1: Clone the Repository

`git clone https://github.com/Ochan-LOKIDORMOI/Summative_assignment-MLOP.git`

# Step 2: Create a Virtual Environment

`
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
`
# Step 3: Install Dependencies
`pip install -r requirements.txt`

# How to Use

# **1. Start the FastAPI Server**
Run the following command in the project directory:

`uvicorn src.api:app --reload `

**2. Start the Flask Application**
Run the Flask application in a new terminal window:

`python -m src.app`
- The application will be accessible at http://127.0.0.1:5000

# **3. Endpoints Overview (FastAPI)**
- **Prediction:** /predict/ Upload an image to predict if the animal is domestic or wild.
- **Model Training:** /train-model/ Upload a zipped folder with labeled images (wild, domestic) to train a new model.
- **Model Listing:** /list-models/ Fetch all available models for prediction.

# **4. Dataset Upload for Retraining (Flask)**
- Navigate to /retrain.html on the Flask application.
- Upload a .zip file with structured datasets.
- View retraining results and performance metrics.

# Deployment
 ## Using Docker

**1. Build the Docker image:**
`docker build -t animal-classification-app .`

**2. Run the Docker container:**
`docker run -p 8000:8000 animal-classification-app`

**3. Use Docker Compose (for Flask-FastAPI integration)**
`docker-compose up`


# **Future Improvements**
- **Optimization:** Enhance training speed and memory efficiency.
- **GPU Support:** Integrate GPU-based model training for improved performance.
- **Extended Functionality:** Support additional animal categories and include more advanced metrics.

# Author
Ochan Denmark LOKIDORMOI
o.lokidormo@alustudent.com

**Enjoy using the Animal Classification System! 🐾**
