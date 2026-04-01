import cv2
import numpy as np
import csv
import random
from pathlib import Path

# --- CONFIGURATION ---
# Chemin vers ton dossier 'dataset' qui contient les dossiers 'signe'
input_dir = Path(r'C:\Users\Administrateur\Documents\hope\dataset')
output_dir = Path(r'C:\Users\Administrateur\Documents\hope\dataset_pro_final')
csv_path = output_dir / "dataset_metadata.csv"
TARGET_COUNT = 45 

output_dir.mkdir(exist_ok=True)

def safe_read_image(file_path):
    """ Lecture robuste pour Windows (gère les accents) """
    try:
        file_bytes = np.fromfile(str(file_path), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except:
        return None

def appliquer_augmentation(image):
    """ Applique les contraintes : Rotation max 10°, Zoom max 0.2 """
    if image is None: return None
    h, w = image.shape[:2]
    
    # 1. Rotation aléatoire entre -10 et 10 degrés
    angle = random.uniform(-10, 10)
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img_aug = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # 2. Zoom aléatoire (0.8 à 1.0 pour simuler un zoom de 20%)
    zoom_factor = random.uniform(0.8, 1.0)
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    img_aug = img_aug[start_y:start_y+new_h, start_x:start_x+new_w]
    
    # Redimensionnement standard pour l'entraînement
    return cv2.resize(img_aug, (224, 224))

def traiter_dataset():
    csv_data = []
    
    # Parcourir les dossiers de signes (ex: maman, travail...)
    for dossier_signe in input_dir.iterdir():
        if not dossier_signe.is_dir(): continue
        
        # Parcourir les dossiers de personnes (ex: emna, eya...)
        for dossier_personne in dossier_signe.iterdir():
            if not dossier_personne.is_dir(): continue
            
            signe_nom = dossier_signe.name
            personne_nom = dossier_personne.name
            
            print(f"🔄 Traitement : {signe_nom} | Individu : {personne_nom}")
            
            # Dossier de sortie organisé par signe
            dest_folder = output_dir / signe_nom
            dest_folder.mkdir(exist_ok=True)
            
            # Récupérer les images existantes de cette personne pour ce signe
            images_sources = [f for f in dossier_personne.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            if not images_sources:
                print(f"⚠️ Aucune image pour {personne_nom} dans {signe_nom}")
                continue

            for i in range(1, TARGET_COUNT + 1):
                # Format du nom : signe-personne-i.jpg
                nouveau_nom = f"{signe_nom}-{personne_nom}-{i}.jpg"
                final_path = dest_folder / nouveau_nom
                
                # Si on a encore des images originales, on les utilise
                if (i - 1) < len(images_sources):
                    img = safe_read_image(images_sources[i-1])
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                else:
                    # Sinon, on prend une image originale au hasard et on l'augmente
                    img_base = safe_read_image(random.choice(images_sources))
                    img = appliquer_augmentation(img_base)
                
                if img is not None:
                    cv2.imwrite(str(final_path), img)
                    csv_data.append([str(final_path), signe_nom, personne_nom])

    # Génération du CSV final
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label_signe', 'individu'])
        writer.writerows(csv_data)

if __name__ == "__main__":
    traiter_dataset()
    print(f"\n✅ Succès ! Ton dataset équilibré est prêt dans : {output_dir}")