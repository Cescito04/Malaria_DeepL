# 🔧 Fix: Worker Timeout sur Render

## Problème

Le worker Gunicorn timeout après ~30 secondes au lieu des 300 secondes configurées. Cela signifie que le Start Command dans Render n'utilise pas la configuration avec timeout.

## Solution

### Étape 1: Vérifier le Start Command dans Render

1. Allez sur votre dashboard Render : https://dashboard.render.com
2. Sélectionnez votre service `malaria-deepl`
3. Cliquez sur **"Settings"** dans le menu de gauche
4. Faites défiler jusqu'à **"Start Command"**
5. Vérifiez le contenu actuel

### Étape 2: Mettre à jour le Start Command

**Remplacez** le Start Command actuel par :

```bash
python -m gunicorn app:app --config gunicorn_config.py --timeout 300 --graceful-timeout 120
```

**OU** si vous préférez sans le fichier de config :

```bash
python -m gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --graceful-timeout 120 --workers 1
```

### Étape 3: Sauvegarder et redéployer

1. Cliquez sur **"Save Changes"**
2. Render redéploiera automatiquement avec le nouveau Start Command
3. Attendez que le déploiement se termine

### Étape 4: Vérifier les logs

Après le redéploiement, vérifiez les logs pour confirmer que le timeout est bien appliqué :

```
[INFO] Starting gunicorn 23.0.0
[INFO] Listening at: http://0.0.0.0:10000
[INFO] Using worker: sync
```

Si vous voyez toujours des timeouts après 30 secondes, le Start Command n'a peut-être pas été mis à jour correctement.

## Alternative: Utiliser le Procfile

Si Render utilise automatiquement le `Procfile`, assurez-vous qu'il contient :

```
web: python -m gunicorn app:app --config gunicorn_config.py --timeout 300 --graceful-timeout 120
```

## Vérification

Pour vérifier que le timeout est bien appliqué, testez une prédiction et surveillez les logs. Si le timeout est de 300 secondes, vous ne devriez plus voir de "WORKER TIMEOUT" après 30 secondes.

## Si le problème persiste

Si le timeout persiste même après avoir mis à jour le Start Command :

1. **Vérifiez les logs de build** pour voir quel Start Command est utilisé
2. **Vérifiez la mémoire** : Le plan gratuit de Render offre 512 MB. Si le modèle est trop lourd, vous pourriez avoir besoin d'un plan supérieur
3. **Considérez un modèle plus léger** : Le modèle VGG peut être trop lourd. Vous pourriez utiliser le modèle "simple" à la place

## Notes

- Le timeout de 300 secondes (5 minutes) devrait être largement suffisant pour une prédiction TensorFlow
- Le warm-up du modèle au démarrage évite le délai de compilation lors de la première vraie prédiction
- L'utilisation de `model.__call__` directement est plus rapide et utilise moins de mémoire que `predict()` ou `predict_on_batch()`

