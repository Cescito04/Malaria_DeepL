# Configuration Gunicorn pour Render
import os
import multiprocessing

# Nombre de workers
workers = 1  # Utiliser 1 worker pour économiser la mémoire (TensorFlow est gourmand)

# Timeout augmenté pour les prédictions TensorFlow (peuvent prendre du temps)
timeout = 120  # 2 minutes

# Bind
bind = "0.0.0.0:{}".format(os.environ.get("PORT", 5001))

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Worker class
worker_class = "sync"

# Preload app pour économiser la mémoire
preload_app = True

# Max requests (recycler les workers après N requêtes pour éviter les fuites mémoire)
max_requests = 1000
max_requests_jitter = 50

