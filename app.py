"""
Application Flask pour la détection du paludisme
Déploie le meilleur modèle CNN pour prédire si une cellule sanguine est infectée
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Créer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle - Détection automatique du meilleur modèle
MODEL_PATH = None
saved_models_dir = 'saved_models'

if os.path.exists(saved_models_dir):
    # Essayer de charger le meilleur modèle depuis les métriques JSON
    metrics_file = os.path.join(saved_models_dir, 'best_model_metrics.json')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            short_name = metrics.get('short_name', 'vgg')
            model_file = os.path.join(saved_models_dir, f'best_model_{short_name}.keras')
            if os.path.exists(model_file):
                MODEL_PATH = model_file
                print(f"✅ Meilleur modèle détecté depuis métriques: {MODEL_PATH}")
                print(f"   Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
        except Exception as e:
            print(f"⚠️  Erreur lors de la lecture des métriques: {e}")
    
    # Si pas trouvé via métriques, chercher tous les fichiers .keras
    if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
        keras_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.keras')]
        if keras_files:
            MODEL_PATH = os.path.join(saved_models_dir, keras_files[0])
            print(f"✅ Modèle trouvé: {MODEL_PATH}")
        else:
            raise FileNotFoundError(f"❌ Aucun modèle .keras trouvé dans {saved_models_dir}")
else:
    raise FileNotFoundError(f"❌ Dossier {saved_models_dir} introuvable")

print(f"Chargement du modèle depuis: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(" Modèle chargé avec succès!")

# Taille d'image attendue par le modèle
IMG_SIZE = (100, 100)  # Ajuster selon la taille utilisée lors de l'entraînement


def allowed_file(filename):
    """Vérifier si le fichier a une extension autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Prétraite une image pour la prédiction
    - Charge l'image
    - Redimensionne à IMG_SIZE
    - Normalise les pixels entre 0 et 1
    - Convertit en RGB si nécessaire
    """
    try:
        # Charger l'image
        img = Image.open(image_path)
        
        # Convertir en RGB si nécessaire
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionner
        img = img.resize(IMG_SIZE)
        
        # Convertir en numpy array et normaliser
        img_array = np.array(img) / 255.0
        
        # Ajouter la dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Erreur lors du prétraitement de l'image: {str(e)}")


def predict_image(image_path):
    """
    Prédit si une cellule est infectée par le paludisme
    Retourne: (probabilité_parasité, probabilité_non_infecté, prédiction)
    """
    try:
        # Prétraiter l'image
        img_array = preprocess_image(image_path)
        
        # Faire la prédiction
        prediction = model.predict(img_array, verbose=0)
        
        # Le modèle retourne une probabilité (binaire: 0 = Uninfected, 1 = Parasitized)
        prob_parasitized = float(prediction[0][0])
        prob_uninfected = 1.0 - prob_parasitized
        
        # Déterminer la classe
        if prob_parasitized > 0.5:
            class_pred = "Parasitized"
            confidence = prob_parasitized
        else:
            class_pred = "Uninfected"
            confidence = prob_uninfected
        
        return {
            'class': class_pred,
            'confidence': round(confidence * 100, 2),
            'prob_parasitized': round(prob_parasitized * 100, 2),
            'prob_uninfected': round(prob_uninfected * 100, 2)
        }
    except Exception as e:
        raise ValueError(f"Erreur lors de la prédiction: {str(e)}")


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour la prédiction via formulaire"""
    try:
        # Vérifier qu'un fichier a été envoyé
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier envoyé'}), 400
        
        file = request.files['file']
        
        # Vérifier qu'un fichier a été sélectionné
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Vérifier l'extension
        if not allowed_file(file.filename):
            return jsonify({'error': f'Extension non autorisée. Extensions autorisées: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Faire la prédiction
        result = predict_image(filepath)
        
        # Supprimer le fichier après prédiction (optionnel)
        # os.remove(filepath)
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """Endpoint API REST pour la prédiction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier envoyé'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'Extension non autorisée'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_image(filepath)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Erreur serveur: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé pour vérifier que le serveur fonctionne"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Serveur Flask de détection du paludisme")
    print("="*60)
    print(f" Modèle: {MODEL_PATH}")
    print(f"Taille d'image: {IMG_SIZE}")
    print(f" Dossier uploads: {UPLOAD_FOLDER}")
    print("="*60)
    print("\n Accédez à l'application sur: http://127.0.0.1:5001")
    print("\n Pour arrêter le serveur, appuyez sur Ctrl+C\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

