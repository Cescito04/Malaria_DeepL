# Malaria_DeepL

## 🔬 Détection du Paludisme par Deep Learning

Projet de deep learning pour la détection automatique du paludisme dans les cellules sanguines à l'aide de réseaux de neurones convolutifs (CNN).

## 📋 Description

Ce projet utilise des modèles de deep learning (CNN) pour classifier automatiquement les cellules sanguines en deux catégories :
- **Parasitized** : Cellules infectées par le paludisme
- **Uninfected** : Cellules non infectées

## 🎯 Fonctionnalités

- ✅ Chargement et prétraitement des données d'images
- ✅ Visualisation des données
- ✅ Normalisation des images
- ✅ Division des données (Train/Validation/Test)
- ✅ Création de 3 modèles CNN (Simple, Profond, Mini-VGG)
- ✅ Entraînement et évaluation des modèles
- ✅ Comparaison des performances
- ✅ Sauvegarde du meilleur modèle
- ✅ Déploiement via serveur Flask avec interface web moderne

## 📊 Performance du modèle

Le meilleur modèle atteint une précision de **95.09%** sur le set de test.

## 🚀 Installation

### Prérequis

```bash
pip install -r requirements.txt
```

### Structure du projet

```
Malaria_DeepL/
├── malaria_detection.ipynb    # Notebook principal avec tout le pipeline
├── app.py                      # Application Flask
├── templates/
│   └── index.html             # Interface web
├── saved_models/              # Modèles sauvegardés
├── cell_images/               # Dataset d'images
│   ├── Parasitized/
│   └── Uninfected/
└── requirements.txt
```

## 💻 Utilisation

### 1. Entraînement du modèle

Exécutez le notebook `malaria_detection.ipynb` pour :
- Charger et prétraiter les données
- Entraîner les modèles CNN
- Évaluer et comparer les performances
- Sauvegarder le meilleur modèle

### 2. Déploiement du serveur Flask

```bash
python app.py
```

Puis accédez à l'interface web sur : `http://127.0.0.1:5001`

## 🌐 API Endpoints

- `GET /` - Interface web
- `POST /predict` - Prédiction via formulaire
- `POST /predict_api` - API REST pour prédiction
- `GET /health` - Vérification de santé du serveur

## 📈 Modèles implémentés

1. **Modèle Simple** : CNN basique avec 2 couches convolutives
2. **Modèle Profond** : CNN avec BatchNormalization et Dropout
3. **Modèle Mini-VGG** : Architecture inspirée de VGG

## 🛠️ Technologies utilisées

- Python 3.x
- TensorFlow/Keras
- Flask
- NumPy, Pandas
- Matplotlib, Seaborn
- PIL (Pillow)
- Scikit-learn

## 📝 Dataset

Le dataset contient environ 27,560 images de cellules sanguines :
- ~13,780 images de cellules infectées (Parasitized)
- ~13,780 images de cellules non infectées (Uninfected)

## 📄 Licence

Ce projet est à des fins éducatives et de recherche.

## 👤 Auteur

Projet développé dans le cadre du Master DSGL.

## 🙏 Remerciements

Merci à la communauté open source pour les outils et bibliothèques utilisés dans ce projet.

