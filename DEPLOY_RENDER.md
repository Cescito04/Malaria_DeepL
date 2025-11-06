# 🚀 Guide de déploiement sur Render

Ce guide vous explique comment déployer l'application Flask de détection du paludisme sur Render.

## 📋 Prérequis

- Un compte GitHub avec le repository `Malaria_DeepL` poussé
- Un compte Render (gratuit disponible sur [render.com](https://render.com))

## 🔧 Étapes de déploiement

### 1. Créer un nouveau service Web sur Render

1. Connectez-vous à votre compte Render
2. Cliquez sur **"New +"** → **"Web Service"**
3. Connectez votre compte GitHub si ce n'est pas déjà fait
4. Sélectionnez le repository **`Malaria_DeepL`**

### 2. Configurer le service

Utilisez les paramètres suivants :

- **Name**: `malaria-deepl` (ou le nom de votre choix)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Plan**: Free (ou un plan payant si vous avez besoin de plus de ressources)

### 3. Variables d'environnement (optionnel)

Render configure automatiquement le port via la variable `PORT`. Vous pouvez ajouter :

- `FLASK_ENV=production` (déjà configuré dans `render.yaml`)

### 4. Déployer

1. Cliquez sur **"Create Web Service"**
2. Render va automatiquement :
   - Cloner votre repository
   - Installer les dépendances (`pip install -r requirements.txt`)
   - Démarrer l'application avec Gunicorn
3. Attendez que le build se termine (peut prendre 5-10 minutes la première fois à cause de TensorFlow)

### 5. Accéder à l'application

Une fois le déploiement terminé, vous obtiendrez une URL comme :
```
https://malaria-deepl.onrender.com
```

## ⚙️ Configuration automatique avec render.yaml

Le fichier `render.yaml` est déjà configuré. Si vous utilisez l'import automatique :

1. Dans Render, cliquez sur **"New +"** → **"Blueprint"**
2. Sélectionnez votre repository
3. Render utilisera automatiquement `render.yaml` pour configurer le service

## 📝 Notes importantes

### Taille du modèle

- Le modèle `best_model_vgg.keras` fait ~57 MB
- GitHub accepte les fichiers jusqu'à 100 MB, donc pas de problème
- Si vous voulez optimiser, vous pouvez utiliser Git LFS pour les gros fichiers

### Temps de build

- Le premier build peut prendre 5-10 minutes à cause de TensorFlow
- Les builds suivants seront plus rapides grâce au cache

### Limites du plan gratuit

- Le service peut "s'endormir" après 15 minutes d'inactivité
- Le premier démarrage après l'endormissement peut prendre 30-60 secondes
- Pour éviter cela, utilisez un plan payant ou un service de "ping" externe

### Mémoire

- TensorFlow nécessite au moins 512 MB de RAM
- Le plan gratuit de Render offre 512 MB, ce qui devrait suffire
- Si vous rencontrez des erreurs de mémoire, passez à un plan supérieur

## 🔍 Vérification du déploiement

Une fois déployé, testez :

1. **Health check**: `https://votre-app.onrender.com/health`
2. **Interface web**: `https://votre-app.onrender.com/`
3. **API**: `POST https://votre-app.onrender.com/predict_api`

## 🐛 Dépannage

### Erreur "Module not found"

- Vérifiez que toutes les dépendances sont dans `requirements.txt`
- Vérifiez les logs de build dans Render

### Erreur "Model not found"

- Vérifiez que le dossier `saved_models/` est bien dans Git
- Vérifiez les logs pour voir le chemin du modèle

### Erreur de mémoire

- Passez à un plan supérieur (au moins 1 GB RAM)
- Ou optimisez le modèle (quantization, pruning)

### Timeout lors du build

- Le build de TensorFlow peut être long
- Augmentez le timeout dans les paramètres Render si nécessaire

## 📚 Ressources

- [Documentation Render](https://render.com/docs)
- [Documentation Flask](https://flask.palletsprojects.com/)
- [Documentation Gunicorn](https://gunicorn.org/)

## ✅ Checklist de déploiement

- [x] Repository GitHub configuré
- [x] `saved_models/` ajouté à Git
- [x] `Procfile` créé
- [x] `render.yaml` créé
- [x] `requirements.txt` mis à jour avec `gunicorn`
- [x] `app.py` configuré pour utiliser le port de Render
- [ ] Service créé sur Render
- [ ] Déploiement réussi
- [ ] Application testée

---

**Bon déploiement ! 🎉**

