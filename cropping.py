import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
model_path = 'best.pt'
input_dir = Path(r'C:\Users\Administrateur\Documents\hope\dataset_pro_final')
output_dir = Path(r'C:\Users\Administrateur\Documents\hope\dataset_cropped_precision')
output_dir.mkdir(parents=True, exist_ok=True)

# Chargement du modèle
model = YOLO(model_path)

def safe_read(path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

def obtenir_crop_precision(img, results, padding=0.1):
    """ Calcule la zone exacte des mains avec une précision maximale """
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        return None

    # 1. Trouver les limites réelles des détections (x_min, y_min, x_max, y_max)
    x1_real = np.min(boxes[:, 0])
    y1_real = np.min(boxes[:, 1])
    x2_real = np.max(boxes[:, 2])
    y2_real = np.max(boxes[:, 3])

    # 2. Calculer la dimension de cette zone
    w_box = x2_real - x1_real
    h_box = y2_real - y1_real

    # 3. Ajouter une marge de sécurité légère (padding)
    x1 = max(0, x1_real - (w_box * padding))
    y1 = max(0, y1_real - (h_box * padding))
    x2 = min(img.shape[1], x2_real + (w_box * padding))
    y2 = min(img.shape[0], y2_real + (h_box * padding))

    # 4. Transformer en CARRE pour éviter la déformation
    new_w = x2 - x1
    new_h = y2 - y1
    side = max(new_w, new_h)
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Calcul des coordonnées du carré final
    final_x1 = int(max(0, cx - side / 2))
    final_y1 = int(max(0, cy - side / 2))
    final_x2 = int(min(img.shape[1], final_x1 + side))
    final_y2 = int(min(img.shape[0], final_y1 + side))

    # Ajustement si on touche les bords de l'image
    crop = img[final_y1:final_y2, final_x1:final_x2]
    
    # Redimensionnement final (toutes les images à 224x224 pour le modèle d'IA)
    if crop.size == 0: return None
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)

def process():
    images = list(input_dir.rglob('*.jpg'))
    print(f"🚀 Début du cropping de précision : {len(images)} images...")
    
    success_count = 0
    for img_path in images:
        img = safe_read(img_path)
        if img is None: continue
        
        # Inférence haute sensibilité
        results = model(img, conf=0.2, verbose=False)
        
        final_crop = obtenir_crop_precision(img, results)
        
        if final_crop is not None:
            # Sauvegarde dans la structure originale
            rel_path = img_path.relative_to(input_dir)
            save_path = output_dir / rel_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(save_path), final_crop)
            success_count += 1
            if success_count % 50 == 0:
                print(f"✅ {success_count} images traitées...")

    print(f"\n✨ Terminé ! {success_count} images ultra-précises dans {output_dir}")

if __name__ == "__main__":
    process()