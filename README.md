Langue des Signes Tunisienne (LST)
Ce projet implémente un système de reconnaissance en temps réel capable de détecter et de classifier des mots spécifiques (et non des lettres isolées) de la Langue des Signes Tunisienne.

Le système utilise une architecture d'intelligence artificielle en cascade pour garantir une précision optimale dans des environnements variés.

🚀 Architecture du Système
Pour maximiser la fiabilité, le projet repose sur deux modèles IA distincts travaillant de concert :

Détecteur de mains (YOLOv8) : Localise les mains dans le flux vidéo et génère une zone d'intérêt (ROI).

Classifieur de Signes (YOLOv8-cls) : Analyse le "crop" (la découpe) de la main pour identifier le mot signé parmi les classes apprises.

🧠 Méthodologie et Entraînement
Ce projet ne se contente pas d'utiliser des modèles existants ; il repose sur un travail de développement IA complet :

Transfer Learning (Apprentissage par transfert) : Les modèles ont été développés via un Fine-tuning de l'architecture YOLOv8. En partant d'un modèle pré-entraîné sur des millions d'images, j'ai spécialisé l'IA sur un dataset propriétaire dédié à la Langue des Signes Tunisienne.

Pipeline de Données personnalisé : * cropping.py : Isolation automatique des mains pour focaliser l'apprentissage sur le signe.

data-augmentation.py : Enrichissement du dataset (variations de lumière, rotations, zooms) pour une meilleure robustesse.

Capacité : Reconnaissance de 26 mots/concepts distincts.

Prétraitement "Global Crop" : Application d'un padding de 30% lors de la détection pour ne jamais couper les mouvements d'amplitude nécessaires aux signes de mots complexes.


⚠️ Note sur les modèles (.pt)
Les fichiers de poids PyTorch (detecteur_main.pt et classifieur_signe.pt) ne sont pas inclus dans ce dépôt pour des raisons de taille.

Utilisation : Pour exécuter le code, placez vos fichiers .pt à la racine ou utilisez les versions .tflite exportées pour le déploiement mobile.
