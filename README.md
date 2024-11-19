# Real vs Fake Image Detection

This project is a Streamlit application that uses a fine-tuned EfficientNet-B0 model to detect if an uploaded image is real or AI-generated.

## Project Overview
The goal of this project is to classify images as either "Real" or "Fake" based on a model trained using EfficientNet-B0. The application is designed for ease of use with a Streamlit interface and performs real-time predictions on uploaded images.

## Project Structure
- **Pipeline Scripts**:
  - `Data Preprocessing Pipeline`: This pipeline prepares the dataset for training, including resizing, normalization, and augmentation.
  - `Model Configuration Pipeline`: This pipeline loads the pre-trained EfficientNet-B0 model, modifies the classifier layer for binary classification, and includes dropout layers.
  - `Training and Validation Pipeline`: Handles model training, validation, and implements early stopping to reduce overfitting.
  - `Evaluation Pipeline`: Evaluates the model's performance on test data and outputs accuracy, precision, recall, and F1 scores.

- **Streamlit Application (`app.py`)**:
  The `app.py` file contains the code to run the Streamlit application. It loads the trained model, allows users to upload images, and outputs a classification label ("Real" or "Fake") with confidence scores.

- **Screenshots**:
  Includes result screenshots demonstrating the application’s functionality and the accuracy of predictions.

## Detailed Descriptions

### Data Preprocessing Pipeline
_Add description here once provided_

### Model Configuration Pipeline
_Add description here once provided_

### Training and Validation Pipeline
_Add description here once provided_

### Evaluation Pipeline
_Add description here once provided_

## Setup Instructions

To run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
