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
Le script de division du dataset organise les images en ensembles d'entraînement (training) et de test (testing) en fonction d'un ratio spécifié (par défaut, 80 % pour l'entraînement et 20 % pour le test). Ce processus permet de créer des ensembles équilibrés et séparés pour l'entraînement et l'évaluation du modèle.

- **Chemins des dossiers d'origine** : Ce script prend les dossiers d'images d'origine, contenant des images étiquetées comme "réelles" (`real_dir`) et "fausses" (`fake_dir`), et les divise en deux ensembles : entraînement et test.
  
- **Création des dossiers** : Le script crée des sous-dossiers spécifiques pour chaque catégorie (`fake_training`, `real_training`, `fake_testing`, `real_testing`) dans les répertoires `training` et `testing`.

- **Fonction `split_data`** : Cette fonction divise les images dans les ensembles d'entraînement et de test en fonction d'un ratio (`split_ratio`) :
  - **Mélange des images** : Les images sont mélangées aléatoirement pour garantir une distribution équilibrée.
  - **Division et copie des fichiers** : Les images sont ensuite copiées dans les dossiers de destination (`train_dest` et `test_dest`) pour les ensembles d'entraînement et de test respectivement.

- **Exécution de la division** : La fonction `split_data` est appelée pour chaque catégorie (réelle et fausse), permettant ainsi de diviser le dataset en deux ensembles finaux.

Après exécution, le script organise le dataset en dossiers de `training` et `testing` prêts à être utilisés pour l’entraînement et l’évaluation du modèle.

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
