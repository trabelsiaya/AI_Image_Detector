Voici un modèle de `README.md` qui décrit votre projet en utilisant les codes fournis et inclut des instructions pour les utilisateurs souhaitant accéder à votre modèle sur Kaggle et l'utiliser dans une application Streamlit :

---

# Détection de Visages Réels et Générés par IA

Ce projet utilise un modèle de classification basé sur `EfficientNet-B0` pour distinguer les visages réels des visages générés par IA. Le modèle a été formé à l'aide de données augmentées pour optimiser sa performance. Le code complet pour la préparation, l'entraînement, et la validation est disponible sur Kaggle, avec le modèle entraîné sauvegardé en tant que fichier `.pth`. 

Les utilisateurs peuvent également utiliser ce modèle dans une application Streamlit (`app.py`), disponible dans ce dépôt, pour faire des prédictions sur de nouvelles images.

## Accéder au Modèle sur Kaggle

Pour accéder au modèle pré-entraîné sur Kaggle :
1. Visitez le lien vers le Kaggle Notebook : `[Lien vers le Notebook Kaggle]`
2. Téléchargez le fichier `best_model.pth` généré par le code dans la section Entraînement.

## Structure du Projet

### 1. Préparation du Dataset

Les étapes ci-dessous décrivent la création du jeu de données pour l'entraînement et le test.

#### Division du Dataset
Le script de division du dataset organise les images en ensembles d'entraînement (training) et de test (testing) en fonction d'un ratio spécifié (par défaut, 80 % pour l'entraînement et 20 % pour le test). Ce processus permet de créer des ensembles équilibrés et séparés pour l'entraînement et l'évaluation du modèle.

Chemins des dossiers d'origine : Ce script prend les dossiers d'images d'origine, contenant des images étiquetées comme "réelles" (real_dir) et "fausses" (fake_dir), et les divise en deux ensembles : entraînement et test.

Création des dossiers : Le script crée des sous-dossiers spécifiques pour chaque catégorie (fake_training, real_training, fake_testing, real_testing) dans les répertoires training et testing.

Fonction split_data : Cette fonction divise les images dans les ensembles d'entraînement et de test en fonction d'un ratio (split_ratio) :

Mélange des images : Les images sont mélangées aléatoirement pour garantir une distribution équilibrée.
Division et copie des fichiers : Les images sont ensuite copiées dans les dossiers de destination (train_dest et test_dest) pour les ensembles d'entraînement et de test respectivement.
Exécution de la division : La fonction split_data est appelée pour chaque catégorie (réelle et fausse), permettant ainsi de diviser le dataset en deux ensembles finaux.

Après exécution, le script organise le dataset en dossiers de training et testing prêts à être utilisés pour l’entraînement et l’évaluation du modèle.


#### Redimensionnement des Images
Ce script redimensionne toutes les images du dataset en une résolution de 224x224 pixels, ce qui est nécessaire pour garantir la compatibilité avec le modèle EfficientNet-B0 utilisé pour la classification. Le redimensionnement améliore également la cohérence des données en entrée du modèle.

Dossiers d'origine : Ce script traite quatre dossiers principaux contenant les images de chaque catégorie (fake_training, fake_testing, real_training, real_testing) dans les répertoires training et testing.

Boucle sur chaque image : Le script parcourt chaque dossier de catégorie et redimensionne toutes les images qu'il contient :

Chargement et redimensionnement : Chaque image est ouverte avec la bibliothèque PIL et redimensionnée à 224x224 pixels.
Remplacement de l'image d'origine : La version redimensionnée de chaque image remplace la version originale directement dans le même dossier.
Gestion des erreurs : Si une image ne peut pas être traitée, une erreur est affichée pour faciliter le débogage.

Une fois le script terminé, toutes les images du dataset sont uniformément redimensionnées et prêtes pour l'étape suivante de traitement et d'entraînement du modèle.

Le dataset est redimensionné pour que toutes les images soient de 224x224 pixels, conforme aux exigences du modèle `EfficientNet-B0`.

#### Augmentation des Données
Ce script augmente le dataset d'entraînement en appliquant deux types de transformations aléatoires sur les images originales : un flip horizontal et un ajustement de couleur (Color Jitter). L'objectif de cette étape est de créer des variations dans les données afin de rendre le modèle plus robuste et d'atténuer le surapprentissage.

Dossiers de training : Ce script cible spécifiquement les dossiers de données d'entraînement (fake_training et real_training).

Transformations :

Redimensionnement : Les images sont d'abord redimensionnées en 224x224 pixels pour maintenir la cohérence avec les autres étapes.
Flip horizontal : Un flip horizontal avec redimensionnement est appliqué sur un sous-ensemble des images (50 images par catégorie), créant des images miroir.
Color Jitter : Cette transformation ajuste aléatoirement la luminosité, le contraste, la saturation et la teinte des images (50 images par catégorie), générant des variations visuelles.
Fonction de sauvegarde :

La fonction save_augmented_image prend une image, applique la transformation spécifiée, puis enregistre l'image augmentée avec un nom de fichier unique qui indique le type d'augmentation appliquée (_flip ou _jitter).
Les nouvelles images augmentées sont sauvegardées directement dans le dossier d'entraînement correspondant.
Après l'exécution de ce script, chaque catégorie d'entraînement contient 100 images supplémentaires (50 par transformation), ce qui augmente la diversité du dataset et renforce les capacités de généralisation du modèle.


### 2. Pipeline de Prétraitement et de Labelisation
Ce script prépare les données d'images pour l'entraînement en appliquant un prétraitement standard et en affectant des étiquettes aux images selon leur catégorie (réel ou faux). Le pipeline facilite le chargement et la transformation des données en utilisant PyTorch.

Classe FaceDataset :

Cette classe hérite de Dataset de PyTorch et prend en charge les images provenant d'un répertoire spécifique.
Chaque instance de FaceDataset est initialisée avec un label (0 pour "real" et 1 pour "fake"), et les images sont lues depuis un dossier spécifique.
Les méthodes implémentées permettent de charger et de transformer chaque image individuellement, en appliquant le label correspondant.
Fonction data_preprocessing_pipeline :

Transformations de base : La fonction définit une série de transformations pour redimensionner les images à 224x224 pixels, les convertir en tenseurs et les normaliser selon les moyennes et écarts-types des canaux RGB (normes de l'ImageNet).
Création des datasets : Deux ensembles de données (real_dataset et fake_dataset) sont créés à partir des images étiquetées. Les transformations sont appliquées au chargement.
Concaténation et DataLoader : Les datasets pour les images "réelles" et "fausses" sont combinés avec ConcatDataset. Le DataLoader crée des lots de données (batchs) et utilise le shuffle pour mélanger les données à chaque époque, facilitant ainsi l'entraînement du modèle.
Étiquetage : Ce pipeline permet un étiquetage automatique des images, avec 0 pour les images "réelles" et 1 pour les images "fausses". Cette labélisation est essentielle pour l’entraînement supervisé du modèle.

Une fois ce pipeline exécuté, les images du dataset sont prêtes à être chargées et traitées de manière cohérente pour l’entraînement du modèle.

### 3. Configuration du Modèle
Ce code permet de tester le DataLoader créé par la pipeline de prétraitement et de labélisation. Il vérifie que les images sont correctement chargées avec les étiquettes correctes, ce qui est essentiel pour garantir que le modèle s'entraîne sur des données bien préparées.

Chargement des Données : Le DataLoader est configuré avec la fonction data_preprocessing_pipeline, qui prend en entrée les dossiers de données d’entraînement pour les images "réelles" et "fausses" et crée des lots (batches) de taille spécifiée (batch_size=8).

Vérification des Étiquettes :

Le script parcourt les images et leurs étiquettes dans le premier lot pour s'assurer que les étiquettes sont correctement affectées :
0 pour "Real" (images réelles).
1 pour "Fake" (images générées).
Pour chaque image dans le premier lot, le code affiche le label associé, permettant de confirmer visuellement que les images et les étiquettes sont cohérentes.
Cette vérification aide à s'assurer que le DataLoader est configuré correctement, avec des images chargées et labélisées de manière appropriée pour l’entraînement.

### 4. Entraînement et Validation


Cette pipeline réalise l'entraînement et la validation du modèle avec :
- Early stopping pour éviter le surapprentissage
- Suivi des métriques de précision, rappel, F1-score, et perte

Le modèle entraîné est sauvegardé sous le nom `best_model.pth`.

### 5. Évaluation Finale

Cette pipeline évalue le modèle sur le jeu de test et affiche les métriques de performance finales.

### 6. Application Streamlit

L'application `app.py` permet de charger une image, d'appliquer le modèle pour détecter si le visage est réel ou généré par IA, et d'afficher le résultat. Assurez-vous de placer le fichier `best_model.pth` dans le même dossier que `app.py`.

## Utiliser l'Application Streamlit

1. Installez les bibliothèques nécessaires :
   ```bash
   pip install torch torchvision streamlit
   ```

2. Placez le fichier `best_model.pth` dans le même répertoire que `app.py`.

3. Exécutez l'application Streamlit :
   ```bash
   streamlit run app.py
   ```

4. Chargez une image dans l'interface Streamlit pour voir si elle est classée comme un visage réel ou généré par IA.

