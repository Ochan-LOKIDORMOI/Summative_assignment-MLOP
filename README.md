# Animal Classification Model

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

Summative_Assignment-MLOP/
├── README.md          # Documentation
├── notebook/          # Jupyter Notebooks for initial experiments
│   └── Ochan_LOKIDORMOI_MLOP_Summative_Assignment.ipynb
├── src/               # Source code for the project
│   ├── preprocessing.py  # Image preprocessing utilities
│   ├── model.py          # ML model definition and training logic
│   └── prediction.py     # Model inference and prediction
├── data/              
│   ├── test/           
│   └── train/      
├── models/           
│   ├── pipe.pkl