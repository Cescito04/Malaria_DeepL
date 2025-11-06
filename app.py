"""
Application Flask pour la détection du paludisme
Déploie le meilleur modèle CNN pour prédire si une cellule sanguine est infectée
"""

import os
import json
import logging
import traceback
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
try:
    # Optimiser TensorFlow pour les prédictions
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modèle chargé avec succès!")
    logger.info(f"Modèle chargé: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    raise

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
        logger.info(f"Début de la prédiction pour: {image_path}")
        
        # Prétraiter l'image
        img_array = preprocess_image(image_path)
        logger.info(f"Image prétraitée, shape: {img_array.shape}")
        
        # Faire la prédiction avec timeout implicite
        # Utiliser predict_on_batch pour être plus rapide
        prediction = model.predict_on_batch(img_array)
        logger.info(f"Prédiction effectuée: {prediction}")
        
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
        
        result = {
            'class': class_pred,
            'confidence': round(confidence * 100, 2),
            'prob_parasitized': round(prob_parasitized * 100, 2),
            'prob_uninfected': round(prob_uninfected * 100, 2)
        }
        
        logger.info(f"Prédiction réussie: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Erreur lors de la prédiction: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise ValueError(error_msg)


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour la prédiction via formulaire"""
    try:
        logger.info("Requête POST reçue sur /predict")
        
        # Vérifier qu'un fichier a été envoyé
        if 'file' not in request.files:
            logger.warning("Aucun fichier dans la requête")
            return jsonify({'error': 'Aucun fichier envoyé'}), 400
        
        file = request.files['file']
        
        # Vérifier qu'un fichier a été sélectionné
        if file.filename == '':
            logger.warning("Nom de fichier vide")
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Vérifier l'extension
        if not allowed_file(file.filename):
            logger.warning(f"Extension non autorisée: {file.filename}")
            return jsonify({'error': f'Extension non autorisée. Extensions autorisées: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Vérifier que le dossier uploads existe
        upload_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        logger.info(f"Sauvegarde du fichier: {filepath}")
        file.save(filepath)
        
        # Vérifier que le fichier existe
        if not os.path.exists(filepath):
            logger.error(f"Le fichier n'a pas été sauvegardé: {filepath}")
            return jsonify({'error': 'Erreur lors de la sauvegarde du fichier'}), 500
        
        # Faire la prédiction
        result = predict_image(filepath)
        
        # Supprimer le fichier après prédiction pour économiser l'espace
        try:
            os.remove(filepath)
            logger.info(f"Fichier supprimé: {filepath}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le fichier: {e}")
        
        return jsonify(result)
    
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        error_msg = f'Erreur serveur: {str(e)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500


@app.route('/predict_api', methods=['POST'])
def predict_api():
    """Endpoint API REST pour la prédiction"""
    try:
        logger.info("Requête POST reçue sur /predict_api")
        
        if 'file' not in request.files:
            logger.warning("Aucun fichier dans la requête")
            return jsonify({'success': False, 'error': 'Aucun fichier envoyé'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Nom de fichier vide")
            return jsonify({'success': False, 'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"Extension non autorisée: {file.filename}")
            return jsonify({'success': False, 'error': f'Extension non autorisée'}), 400
        
        # Vérifier que le dossier uploads existe
        upload_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        logger.info(f"Sauvegarde du fichier: {filepath}")
        file.save(filepath)
        
        # Vérifier que le fichier existe
        if not os.path.exists(filepath):
            logger.error(f"Le fichier n'a pas été sauvegardé: {filepath}")
            return jsonify({'success': False, 'error': 'Erreur lors de la sauvegarde du fichier'}), 500
        
        result = predict_image(filepath)
        
        # Supprimer le fichier après prédiction
        try:
            os.remove(filepath)
            logger.info(f"Fichier supprimé: {filepath}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le fichier: {e}")
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        error_msg = f'Erreur serveur: {str(e)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'error': error_msg}), 500


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
    
    # Port pour Render (utilise la variable d'environnement PORT si disponible)
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"\n Accédez à l'application sur: http://127.0.0.1:{port}")
    print("\n Pour arrêter le serveur, appuyez sur Ctrl+C\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)

