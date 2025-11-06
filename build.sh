#!/bin/bash
# Script de build pour Render

echo "🔧 Mise à jour de pip..."
pip install --upgrade pip || true

echo "📦 Installation des dépendances..."
pip install -r requirements.txt

echo "✅ Vérification de l'installation de gunicorn..."
if ! python -c "import gunicorn; print(f'Gunicorn version: {gunicorn.__version__}')" 2>/dev/null; then
    echo "❌ Gunicorn non trouvé! Réinstallation explicite..."
    pip install --force-reinstall gunicorn>=21.2.0
    echo "✅ Gunicorn réinstallé!"
else
    echo "✅ Gunicorn est installé!"
fi

echo "📋 Liste des packages installés (vérification)..."
pip list | grep -i gunicorn || echo "⚠️  Gunicorn toujours non trouvé dans pip list"

echo "✅ Build terminé!"

