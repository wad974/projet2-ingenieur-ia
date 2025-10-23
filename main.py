# Guide Complet - API Hugging Face pour la Segmentation Vestimentaire
# ModeTrends - Fashion Trend Intelligence

# ============================================================
# 1. IMPORTS ET CONFIGURATION INITIALE
# ============================================================

import os
from dotenv import load_dotenv
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#from tqdm.notebook import tqdm
from tqdm import tqdm
import base64
import io
import time
#import function
from app import get_image_dimensions, create_masks, segment_images_batch, display_segmented_images_batch

# Configuration des chemins et paramètres
image_dir = "jeu_de_donnees/top_influenceurs_2024/IMG/"  # Modifiez selon votre environnement
max_images = 50  # Nombre d'images à traiter
mode_alone = False  #bool vrai = un seul image ou false plusieur image

# IMPORTANT: Remplacez par votre token API personnel
load_dotenv()
api_token = os.getenv('HF_TOKEN')


# Créer le dossier d'images s'il n'existe pas
if not os.path.exists(image_dir):
    os.makedirs(image_dir, exist_ok=True) # ne pas râler si le dossier existe déjà
    print(f"Dossier '{image_dir}' créé. Veuillez y ajouter des images .jpg ou .png.")
else:
    print(f"Dossier '{image_dir}' existant.")

# Vérifier que le token est configuré
if not api_token or api_token == "":
    print("\n❌ ERREUR : Token API non configuré !")
    print("   Solution : export HF_TOKEN=votre_token_hugging_face")
    print("   Ou créez un fichier .env avec : HF_TOKEN=votre_token")
    exit(1)
else:
    print(f"✅ Token API chargé (commence par: {api_token[:7]}...)")

# ============================================================
# 2. CONFIGURATION DE L'API
# ============================================================

# URL du modèle SegFormer B3 Clothes
API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

headers = {
    "Authorization": f"Bearer {api_token}"
}

# Lister les images à traiter
image_paths = [
    os.path.join(image_dir, f) 
    for f in os.listdir(image_dir) 
    #f.lower() verification insensible a la casse (ex: .jpg ou .JPG)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
][:max_images] # slicing garde seulement les X premiers fichiers

if not image_paths:
    print(f"⚠️ Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"✅ {len(image_paths)} image(s) à traiter : {[os.path.basename(p) for p in image_paths]}")



# ============================================================
# 4. SEGMENTATION D'UNE SEULE IMAGE (Test)
# ============================================================

if mode_alone:
    print(f"✅ Mode seul activé !")
    
    if image_paths:
        single_image_path = image_paths[0]
        print(f"\n🔄 Traitement de l'image : {os.path.basename(single_image_path)}")
        
        try:
            # Lire l'image en binaire
            with open(single_image_path, "rb") as f:
                image_data = f.read()
                
            # Déterminer le type de contenu
            ext = single_image_path.lower().split('.')[-1]
            content_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
            
            # Préparer les headers avec le Content-Type
            headers_with_content = headers.copy()
            headers_with_content["Content-Type"] = content_type
            
            # Envoyer la requête à l'API
            # Dependance du service externe HUGGINGFACE
            # Ajout timeout 30s 
            print("🚀 Envoi de la requête à l'API Hugging Face...")
            response = requests.post(API_URL, headers=headers_with_content, data=image_data, timeout=30)
            
            # Vérifier la réponse
            response.raise_for_status()
            
            # Récupérer les résultats
            results = response.json()
            print(f"✅ Segmentation réussie ! {len(results)} classes détectées.")
            
            # Créer le masque de segmentation
            width, height = get_image_dimensions(single_image_path)
            segmentation_mask = create_masks(results, width, height)
            
            # Afficher les résultats
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Image originale
            original_img = Image.open(single_image_path)
            axes[0].imshow(original_img)
            axes[0].set_title("Image Originale", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Masque de segmentation
            axes[1].imshow(segmentation_mask, cmap='tab20')
            axes[1].set_title("Segmentation des Vêtements", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            #plt.savefig(f"resultats/segmentation_batch_{i}.png", bbox_inches='tight')
            
            # Afficher les classes détectées
            detected_classes = set([r['label'] for r in results if r['label'] != 'Background'])
            print(f"\n👔 Classes détectées : {', '.join(sorted(detected_classes))}")
            
            
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ Erreur HTTP : {e}")
            print(f"Réponse : {response.text}")
        #except timeout
        except requests.exceptions.Timeout:
            print("⏱️ Timeout : l'API a mis trop de temps à répondre.")
        except Exception as e:
            print(f"❌ Une erreur est survenue : {e}")
    else:
        print("⚠️ Aucune image à traiter. Vérifiez la configuration de 'image_dir'.")
else :
    print(f"⚠️ Mode seul désactivé ! nombre d'image à traité {max_images}")
    
    # Exécuter le traitement batch
    if image_paths:
        print(f"\n📊 Traitement de {len(image_paths)} image(s) en batch...")
        batch_seg_results = segment_images_batch(image_paths, headers, API_URL)
        print("✅ Traitement en batch terminé.")
    else:
        batch_seg_results = []
        print("⚠️ Aucune image à traiter en batch.")
        
    # ============================================================
    # 6. AFFICHAGE DES RÉSULTATS EN BATCH
    # ============================================================
    
    # Afficher les résultats du batch
    if batch_seg_results:
        display_segmented_images_batch(image_paths, batch_seg_results)
        
        successes = sum(1 for mask in batch_seg_results if mask is not None)
        failures = len(batch_seg_results) - successes
    else:
        print("⚠️ Aucun résultat de segmentation à afficher.")

    # ============================================================
    # 7. RÉSUMÉ ET STATISTIQUES
    # ============================================================
    # ============================================================
    # 8. CONCLUSION
    # ============================================================
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DU TRAITEMENT")
    print("="*60)
    print(f"✅ Segmentations réussies : {successes}/{len(batch_seg_results)}")
    print(f"❌ Échecs : {failures}/{len(batch_seg_results)}")
    print(f"📈 Taux de réussite : {(successes/len(batch_seg_results)*100):.1f}%")
    print("="*60)