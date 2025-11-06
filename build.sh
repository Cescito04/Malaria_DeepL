#!/bin/bash
# Script de build pour Render

set -e  # Arrêter en cas d'erreur

echo "🔧 Mise à jour de pip..."
pip install --upgrade pip

echo "📦 Installation des dépendances..."
pip install -r requirements.txt

echo "✅ Vérification de l'installation de gunicorn..."
python -c "import gunicorn; print(f'Gunicorn version: {gunicorn.__version__}')" || {
    echo "❌ Gunicorn non trouvé! Réinstallation..."
    pip install gunicorn>=21.2.0
}

echo "✅ Build terminé!"

