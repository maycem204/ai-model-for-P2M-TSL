Ce projet implémente un système de reconnaissance de la langue des signes en temps réel, capable de détecter et de classer des mots spécifiques (et non des lettres individuelles) à travers un flux vidéo.

🚀 Architecture du Projet
Le système repose sur un pipeline de détection "en cascade" utilisant deux modèles IA distincts pour maximiser la précision :

Détecteur de mains (YOLOv8) : Localise les mains dans l'image et génère une zone d'intérêt (ROI) englobante.

Classifieur de Signes (YOLOv8-cls) : Analyse le "crop" de la main pour identifier le mot signé parmi les classes apprises.


⚠️ Note sur les modèles (.pt)
Pour des raisons de performance et de stockage, les fichiers de poids PyTorch (.pt) ne sont pas inclus dans ce dépôt.

Instructions : Pour exécuter le code, veuillez placer vos fichiers detecteur_main.pt et classifieur_signe.pt à la racine du projet ou utiliser les versions .tflite exportées pour le déploiement mobile.

🧠 Capacités du Modèle
Contrairement aux systèmes de dactylologie (alphabet), ce modèle a été entraîné pour reconnaître des concepts/mots complets en langue des signes tunisienne .

Nombre de classes : 26 mots/signes distincts.

Prétraitement : Utilisation d'un "Global Crop" avec padding de 30% pour capturer l'amplitude des mouvements nécessaires à la signature de mots.
