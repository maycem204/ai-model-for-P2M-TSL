from ultralytics import YOLO

# On utilise le modèle Medium : excellent équilibre Précision/Vitesse
model = YOLO('yolov8m-cls.pt') 

# Lancer l'entraînement optimisé
results = model.train(
    data=r'C:\Users\Administrateur\Documents\hope\dataset_split',
    epochs=100,              # 100 époques pour laisser le temps au CNN d'apprendre
    imgsz=224,               # Taille standard parfaite pour tes crops
    batch=-1,                # '-1' laisse YOLO choisir la taille idéale pour ta RAM
    patience=20,             # S'arrête si l'accuracy ne monte plus pendant 20 époques
    optimizer='AdamW',       # Le meilleur optimiseur pour la classification
    lr0=0.001,               # Apprentissage progressif pour plus de stabilité
    label_smoothing=0.1,     # Technique Pro pour améliorer la généralisation
    dropout=0.2,             # Évite que le modèle apprenne par cœur (overfitting)
    project='SLR_Project',
    name='Precision_VM_Model'
)

# --- Rapport d'Accuracy final ---
print("\nÉvaluation finale sur le dataset de TEST...")
metrics = model.val(split='test')
print(f"🏆 Accuracy TOP-1 : {metrics.top1*100:.2f}%")