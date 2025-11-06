#!/bin/bash
# Script de build pour Render

set -e  # Arrêter en cas d'erreur

echo "🔧 Mise à jour de pip..."
python3 -m pip install --upgrade pip

echo "📦 Installation des dépendances..."
python3 -m pip install -r requirements.txt

echo "✅ Vérification de l'installation de gunicorn..."
python3 -c "import gunicorn; print(f'Gunicorn version: {gunicorn.__version__}')" || {
    echo "❌ Gunicorn non trouvé! Réinstallation..."
    python3 -m pip install gunicorn>=21.2.0
}

echo "✅ Build terminé!"

