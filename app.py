import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.models import efficientnet_b0  # Import du modèle EfficientNet-B0

# Charger le modèle fine-tuné
@st.cache(allow_output_mutation=True)
def load_model():
    # Charger le modèle EfficientNet-B0 sans pré-entraînement
    model = efficientnet_b0(pretrained=False)
    
    # Adapter la dernière couche pour une classification binaire, comme lors du fine-tuning
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),  # Appliquer le même taux de dropout que pendant l'entraînement
        nn.Linear(model.classifier[1].in_features, 2)
    )
    
    # Charger les poids fine-tunés à partir de votre fichier de modèle
    fine_tuned_weights_path = r"C:/Users/AsusZenbook\Desktop\deepfake\app\best_model.pth"  # Assurez-vous que le chemin est correct
    try:
        model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=torch.device('cpu')))
        print("Modèle fine-tuné chargé avec succès depuis le fichier local.")
    except FileNotFoundError:
        print("Fichier de modèle fine-tuné non trouvé. Assurez-vous que le chemin est correct.")
    
    model.eval()  # Mettre le modèle en mode évaluation
    return model

# Fonction de prédiction
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Ajouter la dimension batch
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return "Real" if predicted.item() == 0 else "Fake"

# Interface utilisateur avec Streamlit
st.title("Image Real or Fake Detector")
st.write("Téléchargez une image pour savoir si elle est réelle ou générée.")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image Téléchargée", use_column_width=True)
    st.write("Classification en cours...")
    
    # Charger le modèle fine-tuné
    model = load_model()
    
    # Prédire l'image
    label = predict_image(image, model)
    st.write(f"Résultat : **{label}**")
