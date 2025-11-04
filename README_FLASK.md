# 🚀 Déploiement du Modèle de Détection du Paludisme avec Flask

## 📋 Description

Application Flask pour déployer le modèle de deep learning de détection du paludisme. L'application permet de prédire si une cellule sanguine est infectée par le paludisme via une interface web intuitive.

## 🔧 Installation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Vérifier que le modèle est sauvegardé

Le modèle doit être sauvegardé dans le dossier `saved_models/` :
- Format `.keras` : `saved_models/best_model_{nom}.keras`
- Fichier de métriques : `saved_models/best_model_metrics.json` (optionnel mais recommandé)

L'application détecte automatiquement le meilleur modèle depuis le fichier de métriques ou utilise le premier fichier `.keras` trouvé.

## 🎯 Utilisation

### Lancer le serveur

```bash
python app.py
```

Le serveur démarre sur `http://127.0.0.1:5000`

### Accéder à l'application

Ouvrez votre navigateur et accédez à :
```
http://127.0.0.1:5000
```

## 📡 Endpoints API

### 1. Interface Web
- **URL** : `GET /`
- **Description** : Interface web pour uploader et analyser des images

### 2. Prédiction (Formulaire)
- **URL** : `POST /predict`
- **Méthode** : POST
- **Body** : `multipart/form-data` avec champ `file`
- **Réponse** : JSON avec prédiction
```json
{
  "class": "Parasitized",
  "confidence": 95.09,
  "prob_parasitized": 95.09,
  "prob_uninfected": 4.91
}
```

### 3. Prédiction (API REST)
- **URL** : `POST /predict_api`
- **Méthode** : POST
- **Body** : `multipart/form-data` avec champ `file`
- **Réponse** : JSON avec statut et prédiction
```json
{
  "success": true,
  "prediction": {
    "class": "Parasitized",
    "confidence": 95.09,
    "prob_parasitized": 95.09,
    "prob_uninfected": 4.91
  }
}
```

### 4. Health Check
- **URL** : `GET /health`
- **Description** : Vérifie que le serveur fonctionne et que le modèle est chargé
- **Réponse** :
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "saved_models/best_model_vgg.keras"
}
```

## 📝 Exemple d'utilisation avec cURL

```bash
# Health check
curl http://127.0.0.1:5000/health

# Prédiction
curl -X POST -F "file=@path/to/image.png" http://127.0.0.1:5000/predict
```

## 📝 Exemple d'utilisation avec Python

```python
import requests

# Health check
response = requests.get('http://127.0.0.1:5000/health')
print(response.json())

# Prédiction
files = {'file': open('image.png', 'rb')}
response = requests.post('http://127.0.0.1:5000/predict', files=files)
print(response.json())
```

## 🎨 Format des images supportés

- PNG
- JPG/JPEG
- GIF
- BMP

Taille maximale : 16 MB

## 📊 Format de sortie

L'application retourne :
- **class** : "Parasitized" ou "Uninfected"
- **confidence** : Niveau de confiance (0-100%)
- **prob_parasitized** : Probabilité d'être infecté (0-100%)
- **prob_uninfected** : Probabilité de ne pas être infecté (0-100%)

## 🔒 Sécurité

- Validation des extensions de fichiers
- Limite de taille de fichier (16 MB)
- Nettoyage des noms de fichiers (secure_filename)

## 📂 Structure des fichiers

```
.
├── app.py              # Application Flask principale
├── requirements.txt   # Dépendances Python
├── templates/          # Templates HTML
│   └── index.html     # Interface web
├── saved_models/       # Modèles sauvegardés
│   ├── best_model_*.keras
│   └── best_model_metrics.json
└── uploads/           # Images uploadées (créé automatiquement)
```

## 🐛 Dépannage

### Erreur : "Aucun modèle .keras trouvé"
- Vérifiez que le modèle est sauvegardé dans `saved_models/`
- Exécutez la cellule de sauvegarde du modèle dans le notebook

### Erreur : "Extension non autorisée"
- Vérifiez que le fichier est bien une image (PNG, JPG, JPEG, GIF, BMP)

### Erreur : "Fichier trop volumineux"
- Réduisez la taille de l'image (max 16 MB)

## 📚 Technologies utilisées

- **Flask** : Framework web Python
- **TensorFlow/Keras** : Deep Learning
- **Pillow (PIL)** : Traitement d'images
- **NumPy** : Calculs numériques

## 🚀 Déploiement en production

Pour déployer en production, considérez :

1. **Gunicorn** pour servir l'application :
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Nginx** comme reverse proxy

3. **HTTPS** pour la sécurité

4. **Variables d'environnement** pour la configuration

## 📞 Support

Pour toute question ou problème, consultez le notebook `malaria_detection.ipynb` pour plus de détails sur le modèle et son entraînement.

