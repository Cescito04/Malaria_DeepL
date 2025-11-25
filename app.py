"""
Application Flask pour la détection du paludisme
Déploie le meilleur modèle CNN pour prédire si une cellule sanguine est infectée
"""

import os
import json
import logging
import traceback
import secrets
import requests
from datetime import datetime, timedelta
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from PIL import Image
import tensorflow as tf

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
UPLOAD_FOLDER = 'uploads'
HISTORY_FILE = 'history.json'
USERS_FILE = 'users.json'
RESET_TOKENS_FILE = 'reset_tokens.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max
MAX_HISTORY_ENTRIES = 100  # Limiter l'historique à 100 entrées

# Configuration Brevo (Sendinblue)
BREVO_API_KEY = os.environ.get('BREVO_API_KEY')
if not BREVO_API_KEY:
    logger.warning("BREVO_API_KEY environment variable not set. Email functionality will be disabled.")
BREVO_SENDER_EMAIL = os.environ.get('BREVO_SENDER_EMAIL', 'bocoumabdoulaye988@gmail.com')
BREVO_SENDER_NAME = os.environ.get('BREVO_SENDER_NAME', 'AI Malaria Detection')
BREVO_API_URL = 'https://api.brevo.com/v3/smtp/email'

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
    # Optimiser TensorFlow pour les prédictions et réduire la consommation mémoire
    # Limiter les threads pour économiser la mémoire
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    # Désactiver le GPU (pas disponible sur Render de toute façon)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except:
        pass  # Pas de GPU disponible, c'est normal
    
    # Désactiver les optimisations gourmandes en mémoire
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Désactiver OneDNN pour économiser la mémoire
    
    # Charger le modèle
    logger.info("Chargement du modèle en cours...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Warm-up: faire une prédiction factice pour initialiser le modèle
    # Cela évite le délai de compilation lors de la première vraie prédiction
    logger.info("Warm-up du modèle (prédiction factice)...")
    try:
        dummy_input = np.zeros((1, 100, 100, 3), dtype=np.float32)
        _ = model(dummy_input, training=False)
        logger.info("✅ Warm-up réussi!")
    except Exception as e:
        logger.warning(f"Warm-up échoué (non critique): {e}")
    
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


# ==================== AUTHENTICATION ====================

def load_users():
    """Charger les utilisateurs depuis le fichier JSON"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des utilisateurs: {e}")
        return {}


def save_users(users):
    """Sauvegarder les utilisateurs dans le fichier JSON"""
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des utilisateurs: {e}")


def login_required(f):
    """Décorateur pour protéger les routes nécessitant une authentification"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentification requise', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    return decorated_function


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Inscription d'un nouvel utilisateur"""
    # Si déjà connecté, rediriger vers la page d'accueil
    if request.method == 'GET' and 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username', '').strip()
            email = data.get('email', '').strip()
            password = data.get('password', '')
            
            # Validation
            if not username or not email or not password:
                return jsonify({'success': False, 'error': 'Tous les champs sont requis'}), 400
            
            if len(username) < 3:
                return jsonify({'success': False, 'error': 'Le nom d\'utilisateur doit contenir au moins 3 caractères'}), 400
            
            if len(password) < 6:
                return jsonify({'success': False, 'error': 'Le mot de passe doit contenir au moins 6 caractères'}), 400
            
            # Charger les utilisateurs
            users = load_users()
            
            # Vérifier si l'utilisateur existe déjà
            if username in users:
                return jsonify({'success': False, 'error': 'Ce nom d\'utilisateur est déjà pris'}), 400
            
            # Vérifier si l'email existe déjà
            for user in users.values():
                if user.get('email') == email:
                    return jsonify({'success': False, 'error': 'Cet email est déjà utilisé'}), 400
            
            # Créer le nouvel utilisateur
            users[username] = {
                'username': username,
                'email': email,
                'password': generate_password_hash(password),
                'created_at': datetime.now().isoformat(),
                'analyses_count': 0
            }
            
            save_users(users)
            logger.info(f"Nouvel utilisateur créé: {username}")
            
            # CONNEXION AUTOMATIQUE après inscription
            session['user_id'] = username
            session['user_email'] = email
            logger.info(f"Utilisateur automatiquement connecté après inscription: {username}")
            
            return jsonify({
                'success': True,
                'message': 'Compte créé avec succès. Vous êtes maintenant connecté.',
                'user': {
                    'username': username,
                    'email': email
                },
                'redirect': '/'  # Redirection vers la page d'accueil
            })
            
        except Exception as e:
            logger.error(f"Erreur lors de l'inscription: {e}")
            return jsonify({'success': False, 'error': 'Erreur serveur lors de l\'inscription'}), 500
    
    return render_template('auth.html', mode='register')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Connexion d'un utilisateur"""
    # Si déjà connecté, rediriger vers la page d'accueil
    if request.method == 'GET' and 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username', '').strip()
            password = data.get('password', '')
            
            # Validation
            if not username or not password:
                return jsonify({'success': False, 'error': 'Nom d\'utilisateur et mot de passe requis'}), 400
            
            # Charger les utilisateurs
            users = load_users()
            
            # Vérifier l'utilisateur
            if username not in users:
                return jsonify({'success': False, 'error': 'Nom d\'utilisateur ou mot de passe incorrect'}), 401
            
            user = users[username]
            
            # Vérifier le mot de passe
            if not check_password_hash(user['password'], password):
                return jsonify({'success': False, 'error': 'Nom d\'utilisateur ou mot de passe incorrect'}), 401
            
            # Créer la session
            session['user_id'] = username
            session['user_email'] = user.get('email', '')
            logger.info(f"Utilisateur connecté: {username}")
            
            return jsonify({
                'success': True,
                'message': 'Connexion réussie',
                'user': {
                    'username': username,
                    'email': user.get('email', '')
                },
                'redirect': '/'
            })
            
        except Exception as e:
            logger.error(f"Erreur lors de la connexion: {e}")
            return jsonify({'success': False, 'error': 'Erreur serveur lors de la connexion'}), 500
    
    return render_template('auth.html', mode='login')


@app.route('/logout', methods=['POST'])
def logout():
    """Déconnexion de l'utilisateur"""
    username = session.get('user_id', 'Unknown')
    session.clear()
    logger.info(f"Utilisateur déconnecté: {username}")
    return jsonify({'success': True, 'message': 'Déconnexion réussie'})


@app.route('/auth/check', methods=['GET'])
def check_auth():
    """Vérifier si l'utilisateur est authentifié"""
    if 'user_id' in session:
        users = load_users()
        username = session['user_id']
        if username in users:
            return jsonify({
                'authenticated': True,
                'user': {
                    'username': username,
                    'email': session.get('user_email', '')
                }
            })
    
    return jsonify({'authenticated': False})


# ==================== MOT DE PASSE OUBLIÉ ====================

def load_reset_tokens():
    """Charger les tokens de réinitialisation depuis le fichier JSON"""
    try:
        if os.path.exists(RESET_TOKENS_FILE):
            with open(RESET_TOKENS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"Erreur lors du chargement des tokens: {e}")
        return {}


def save_reset_tokens(tokens):
    """Sauvegarder les tokens de réinitialisation dans le fichier JSON"""
    try:
        with open(RESET_TOKENS_FILE, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des tokens: {e}")


def send_email_via_brevo(to_email, to_name, subject, html_content):
    """Envoyer un email via l'API Brevo"""
    if not BREVO_API_KEY:
        logger.error("BREVO_API_KEY not set. Cannot send email.")
        return False
    
    try:
        headers = {
            'accept': 'application/json',
            'api-key': BREVO_API_KEY,
            'content-type': 'application/json'
        }
        
        payload = {
            'sender': {
                'name': BREVO_SENDER_NAME,
                'email': BREVO_SENDER_EMAIL
            },
            'to': [
                {
                    'email': to_email,
                    'name': to_name
                }
            ],
            'subject': subject,
            'htmlContent': html_content
        }
        
        response = requests.post(BREVO_API_URL, headers=headers, json=payload)
        
        if response.status_code == 201:
            logger.info(f"Email envoyé avec succès à {to_email}")
            return True
        else:
            logger.error(f"Erreur lors de l'envoi de l'email: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email via Brevo: {e}")
        return False


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Demander la réinitialisation du mot de passe"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            email = data.get('email', '').strip().lower()
            
            if not email:
                return jsonify({'success': False, 'error': 'Email requis'}), 400
            
            # Charger les utilisateurs
            users = load_users()
            
            # Trouver l'utilisateur par email
            user = None
            username = None
            for uname, u in users.items():
                if u.get('email', '').lower() == email:
                    user = u
                    username = uname
                    break
            
            # Toujours retourner succès pour éviter l'énumération d'emails
            if not user:
                logger.warning(f"Tentative de réinitialisation pour email inexistant: {email}")
                return jsonify({
                    'success': True,
                    'message': 'Si cet email existe, un lien de réinitialisation a été envoyé.'
                })
            
            # Générer un token de réinitialisation
            token = secrets.token_urlsafe(32)
            expires_at = (datetime.now() + timedelta(hours=1)).isoformat()  # Token valide 1 heure
            
            # Sauvegarder le token
            tokens = load_reset_tokens()
            tokens[token] = {
                'username': username,
                'email': email,
                'expires_at': expires_at,
                'created_at': datetime.now().isoformat()
            }
            save_reset_tokens(tokens)
            
            # Générer le lien de réinitialisation
            reset_url = request.host_url.rstrip('/') + url_for('reset_password', token=token)
            
            # Contenu de l'email
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                    .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }}
                    .button {{ display: inline-block; background: #6366f1; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                    .footer {{ text-align: center; margin-top: 20px; color: #6b7280; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🔬 AI Malaria Detection</h1>
                        <p>Réinitialisation de mot de passe</p>
                    </div>
                    <div class="content">
                        <p>Bonjour {user.get('username', 'Utilisateur')},</p>
                        <p>Vous avez demandé la réinitialisation de votre mot de passe.</p>
                        <p>Cliquez sur le bouton ci-dessous pour réinitialiser votre mot de passe :</p>
                        <p style="text-align: center;">
                            <a href="{reset_url}" class="button">Réinitialiser mon mot de passe</a>
                        </p>
                        <p>Ou copiez ce lien dans votre navigateur :</p>
                        <p style="word-break: break-all; color: #6366f1;">{reset_url}</p>
                        <p><strong>Ce lien est valide pendant 1 heure.</strong></p>
                        <p>Si vous n'avez pas demandé cette réinitialisation, ignorez cet email.</p>
                    </div>
                    <div class="footer">
                        <p>© 2025 AI Malaria Detection - Tous droits réservés</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Envoyer l'email
            if send_email_via_brevo(email, user.get('username', 'Utilisateur'), 'Réinitialisation de votre mot de passe', html_content):
                logger.info(f"Email de réinitialisation envoyé à {email}")
                return jsonify({
                    'success': True,
                    'message': 'Si cet email existe, un lien de réinitialisation a été envoyé.'
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Erreur lors de l\'envoi de l\'email. Veuillez réessayer plus tard.'
                }), 500
            
        except Exception as e:
            logger.error(f"Erreur lors de la demande de réinitialisation: {e}")
            return jsonify({'success': False, 'error': 'Erreur serveur'}), 500
    
    return render_template('auth.html', mode='forgot_password')


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Réinitialiser le mot de passe avec un token"""
    if request.method == 'GET':
        # Vérifier si le token est valide
        tokens = load_reset_tokens()
        
        if token not in tokens:
            return render_template('auth.html', mode='reset_password', error='Token invalide ou expiré', token=token)
        
        token_data = tokens[token]
        expires_at = datetime.fromisoformat(token_data['expires_at'])
        
        if datetime.now() > expires_at:
            # Supprimer le token expiré
            del tokens[token]
            save_reset_tokens(tokens)
            return render_template('auth.html', mode='reset_password', error='Token expiré. Veuillez demander un nouveau lien.', token=None)
        
        return render_template('auth.html', mode='reset_password', token=token)
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            new_password = data.get('password', '')
            confirm_password = data.get('confirm_password', '')
            
            # Validation
            if not new_password or not confirm_password:
                return jsonify({'success': False, 'error': 'Tous les champs sont requis'}), 400
            
            if len(new_password) < 6:
                return jsonify({'success': False, 'error': 'Le mot de passe doit contenir au moins 6 caractères'}), 400
            
            if new_password != confirm_password:
                return jsonify({'success': False, 'error': 'Les mots de passe ne correspondent pas'}), 400
            
            # Vérifier le token
            tokens = load_reset_tokens()
            
            if token not in tokens:
                return jsonify({'success': False, 'error': 'Token invalide ou expiré'}), 400
            
            token_data = tokens[token]
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            
            if datetime.now() > expires_at:
                del tokens[token]
                save_reset_tokens(tokens)
                return jsonify({'success': False, 'error': 'Token expiré. Veuillez demander un nouveau lien.'}), 400
            
            # Mettre à jour le mot de passe
            username = token_data['username']
            users = load_users()
            
            if username not in users:
                return jsonify({'success': False, 'error': 'Utilisateur non trouvé'}), 404
            
            users[username]['password'] = generate_password_hash(new_password)
            save_users(users)
            
            # Supprimer le token utilisé
            del tokens[token]
            save_reset_tokens(tokens)
            
            logger.info(f"Mot de passe réinitialisé pour: {username}")
            
            return jsonify({
                'success': True,
                'message': 'Mot de passe réinitialisé avec succès. Vous pouvez maintenant vous connecter.',
                'redirect': '/login'
            })
            
        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation du mot de passe: {e}")
            return jsonify({'success': False, 'error': 'Erreur serveur'}), 500


def load_history(user_history_file=None):
    """Charger l'historique depuis le fichier JSON"""
    if user_history_file is None:
        username = session.get('user_id', 'anonymous')
        user_history_file = f'history_{username}.json'
    
    try:
        if os.path.exists(user_history_file):
            with open(user_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                if isinstance(history, list):
                    return history
        return []
    except Exception as e:
        logger.warning(f"Erreur lors du chargement de l'historique: {e}")
        return []


def save_history(history, user_history_file=None):
    """Sauvegarder l'historique dans le fichier JSON"""
    if user_history_file is None:
        username = session.get('user_id', 'anonymous')
        user_history_file = f'history_{username}.json'
    
    try:
        # Limiter à MAX_HISTORY_ENTRIES entrées (garder les plus récentes)
        if len(history) > MAX_HISTORY_ENTRIES:
            history = history[-MAX_HISTORY_ENTRIES:]
        
        with open(user_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'historique: {e}")


def add_to_history(filename, result):
    """Ajouter une entrée à l'historique"""
    try:
        # Utiliser un fichier d'historique par utilisateur
        username = session.get('user_id', 'anonymous')
        user_history_file = f'history_{username}.json'
        
        history = load_history(user_history_file)
        
        entry = {
            'id': len(history) + 1,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'class': result['class'],
            'confidence': result['confidence'],
            'prob_parasitized': result['prob_parasitized'],
            'prob_uninfected': result['prob_uninfected']
        }
        
        history.append(entry)
        save_history(history, user_history_file)
        logger.info(f"Entrée ajoutée à l'historique: {entry['id']}")
        
        # Mettre à jour le compteur d'analyses de l'utilisateur
        if 'user_id' in session:
            users = load_users()
            username = session['user_id']
            if username in users:
                users[username]['analyses_count'] = users[username].get('analyses_count', 0) + 1
                save_users(users)
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout à l'historique: {e}")


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
        
        # Faire la prédiction avec optimisations mémoire
        # Utiliser __call__ directement (plus rapide et moins de mémoire)
        logger.info("Début de la prédiction TensorFlow...")
        
        # Convertir en tensor si nécessaire
        if not isinstance(img_array, tf.Tensor):
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        else:
            img_tensor = img_array
        
        # Utiliser __call__ directement (plus rapide et moins de mémoire)
        # Le warm-up au démarrage évite le délai de compilation
        prediction = model(img_tensor, training=False)
        
        # Convertir en numpy si nécessaire
        if hasattr(prediction, 'numpy'):
            prediction = prediction.numpy()
        elif isinstance(prediction, np.ndarray):
            pass  # Déjà en numpy
        else:
            prediction = np.array(prediction)
        
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
    """Page d'accueil - Redirige vers login si non authentifié"""
    # Vérifier si l'utilisateur est connecté
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Charger les métriques du modèle pour les afficher dans la page
    metrics = {}
    metrics_file = os.path.join(saved_models_dir, 'best_model_metrics.json')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Impossible de charger les métriques: {e}")
    
    # Récupérer les informations de l'utilisateur
    user = None
    users = load_users()
    username = session['user_id']
    if username in users:
        user_data = users[username]
        user = {
            'username': username,
            'email': user_data.get('email', '')
        }
    
    return render_template('index.html', metrics=metrics, user=user)


@app.route('/predict', methods=['POST'])
@login_required
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
        
        # Ajouter à l'historique
        add_to_history(filename, result)
        
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
        
        # Ajouter à l'historique
        add_to_history(filename, result)
        
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


@app.route('/history', methods=['GET'])
@login_required
def get_history():
    """Récupérer l'historique des analyses"""
    try:
        limit = request.args.get('limit', type=int)
        history = load_history()
        
        # Inverser pour avoir les plus récentes en premier
        history.reverse()
        
        # Limiter si demandé
        if limit and limit > 0:
            history = history[:limit]
        
        return jsonify({
            'success': True,
            'count': len(history),
            'history': history
        })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/history/clear', methods=['POST'])
@login_required
def clear_history():
    """Effacer l'historique"""
    try:
        save_history([])
        logger.info("Historique effacé")
        return jsonify({
            'success': True,
            'message': 'Historique effacé avec succès'
        })
    except Exception as e:
        logger.error(f"Erreur lors de l'effacement de l'historique: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
@login_required
def get_stats():
    """Récupérer les statistiques globales"""
    try:
        history = load_history()
        
        # Calculer les statistiques
        total_analyses = len(history)
        parasitized_count = sum(1 for item in history if item.get('class') == 'Parasitized')
        uninfected_count = sum(1 for item in history if item.get('class') == 'Uninfected')
        
        avg_confidence = 0
        if total_analyses > 0:
            avg_confidence = sum(item.get('confidence', 0) for item in history) / total_analyses
        
        # Statistiques par jour (7 derniers jours)
        from datetime import datetime, timedelta
        last_7_days = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).date()
            count = sum(1 for item in history 
                       if datetime.fromisoformat(item['timestamp']).date() == date)
            last_7_days.append({
                'date': date.isoformat(),
                'count': count
            })
        
        return jsonify({
            'success': True,
            'stats': {
                'total_analyses': total_analyses,
                'parasitized_count': parasitized_count,
                'uninfected_count': uninfected_count,
                'avg_confidence': round(avg_confidence, 2),
                'last_7_days': last_7_days
            }
        })
    except Exception as e:
        logger.error(f"Erreur lors du calcul des statistiques: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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

