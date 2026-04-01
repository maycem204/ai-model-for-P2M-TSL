import csv
from pathlib import Path

# --- CONFIGURATION ---
# Le dossier où se trouvent tes images générées (45 par personne par signe)
dataset_final_dir = Path(r'C:\Users\Administrateur\Documents\hope\dataset_pro_final')
csv_output_path = dataset_final_dir / "metadata_final.csv"

def generer_csv_final(base_path, output_csv):
    data_rows = []
    # Extensions d'images à inclure
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    print(f"Démarrage du scan de : {base_path}")
    
    # On parcourt tous les fichiers dans tous les sous-dossiers
    for fichier in base_path.rglob('*'):
        if fichier.is_file() and fichier.suffix.lower() in valid_extensions:
            # Nom du fichier (ex: travail-emna-1.jpg)
            nom_image = fichier.name
            
            # Chemin complet (as_posix() pour éviter les problèmes de \ Windows)
            chemin_complet = fichier.absolute().as_posix()
            
            # Optionnel : Extraire le label depuis le nom du fichier
            # Puisque ton nom est signe-personne-i, on split par le tiret '-'
            label = nom_image.split('-')[0]
            
            data_rows.append([nom_image, chemin_complet, label])

    # Écriture du fichier CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # En-tête : Nom, Chemin, et le Label extrait pour faciliter l'IA
        writer.writerow(['image_name', 'path', 'label'])
        writer.writerows(data_rows)

if __name__ == "__main__":
    if dataset_final_dir.exists():
        generer_csv_final(dataset_final_dir, csv_output_path)
        print(f"✅ Succès ! Fichier CSV créé : {csv_output_path}")
        # Petit résumé
        with open(csv_output_path, 'r') as f:
            lignes = len(f.readlines()) - 1
            print(f"📊 Nombre total d'images indexées : {lignes}")
    else:
        print(f"❌ Erreur : Le dossier {dataset_final_dir} n'existe pas.")