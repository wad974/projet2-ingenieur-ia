# Guide Complet - API Hugging Face pour la Segmentation Vestimentaire
# ModeTrends - Fashion Trend Intelligence

# ============================================================
# CLASS MAPPING & COLORMAPS
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
import cv2

# ============================================================
# 3. FONCTIONS UTILITAIRES (Décodage des masques)
# ============================================================

CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

# Colormap personnalisé pour 4 classes (valeurs 1 à 4) en BGR
custom_colormap = {
    1: (0, 255, 255),   # Jaune - Hat
    2: (0, 165, 255),   # Orange - Hair
    3: (255, 0, 255),   # Magenta - Sunglasses
    4: (0, 0, 255),     # Rouge - Upper-clothes
    5: (255, 255, 0),   # Cyan - Skirt
    6: (0, 255, 0),     # Vert - Pants
    7: (255, 0, 0),     # Bleu - Dress
    8: (128, 0, 128),   # Violet - Belt
    9: (0, 255, 255),   # Jaune - Left-shoe
    10: (255, 140, 0),  # Orange foncé - Right-shoe
    11: (200, 180, 140), # Beige - Face
    12: (200, 180, 140), # Beige - Left-leg
    13: (200, 180, 140), # Beige - Right-leg
    14: (200, 180, 140), # Beige - Left-arm
    15: (200, 180, 140), # Beige - Right-arm
    16: (0, 128, 255),  # Bleu clair - Bag
    17: (255, 20, 147)  # Rose - Scarf
}

def add_legend(image, legend, start_x=10, start_y=10, box_size=15, spacing=5):
    """
    Ajoute une légende sur l'image.
    Pour chaque label, dessine un rectangle de la couleur correspondante et le texte associé.
    """
    img_with_legend = image.copy()
    y = start_y
    for label, text in legend.items():
        # Récupération de la couleur du label
        color = custom_colormap.get(int(label), (255, 255, 255))
        # Dessin d'un petit rectangle rempli
        cv2.rectangle(img_with_legend, (start_x, y), (start_x + box_size, y + box_size), color, -1)
        # Ajout du texte à droite du rectangle
        cv2.putText(img_with_legend, text, (start_x + box_size + spacing, y + box_size - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += box_size + spacing
    return img_with_legend
    
# 3. Fonctions pour coloriser le masque et ajouter la légende
def colorize_mask(mask, colormap):
    """
    Applique le colormap personnalisé au masque.
    Pour chaque pixel, s'il correspond à un label défini dans colormap,
    la couleur correspondante est assignée.
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        colored_mask[mask == label] = color
    return colored_mask


def get_image_dimensions(img_path):
    """Récupère les dimensions d'une image."""
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    """Décode un masque encodé en base64."""
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Prendre le premier canal si RGB
    
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)

def create_masks(results, width, height):
    """Combine les masques de plusieurs classes en un seul masque de segmentation."""
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Traiter d'abord les masques non-Background
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Ignorer Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id
    
    # Traiter Background en dernier
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            combined_mask[mask_array > 0] = 0
    
    return combined_mask


# ============================================================
# 5. SEGMENTATION DE PLUSIEURS IMAGES (Batch)
# ============================================================

def segment_images_batch(list_of_image_paths, headers, API_URL):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face.
    
    Args:
        list_of_image_paths (list): Liste des chemins vers les images.
    
    Returns:
        list: Liste des masques de segmentation (tableaux NumPy).
    """
    batch_segmentations = []
    
    for img_path in tqdm(list_of_image_paths, desc="Segmentation des images"):
        try:
            # Lire l'image
            with open(img_path, "rb") as f:
                image_data = f.read()
            
            # Déterminer le Content-Type
            ext = img_path.lower().split('.')[-1]
            content_type = "image/jpeg" if ext in ['jpg', 'jpeg'] else "image/png"
            
            # Préparer les headers
            headers_with_content = headers.copy()
            headers_with_content["Content-Type"] = content_type
            
            # Envoyer la requête à l'API
            # Dependance du service externe HUGGINGFACE
            # Ajout timeout 30s 
            print("🚀 Envoi de la requête à l'API Hugging Face...")
            response = requests.post(API_URL, headers=headers_with_content, data=image_data, timeout=30)
            response.raise_for_status()
            
            # Traiter les résultats
            results = response.json()
            width, height = get_image_dimensions(img_path)
            segmentation_mask = create_masks(results, width, height)
            
            batch_segmentations.append(segmentation_mask)
            
            # Pause pour éviter de surcharger l'API
            time.sleep(1)
            
        except Exception as e:
            print(f"\n❌ Erreur pour {os.path.basename(img_path)}: {e}")
            batch_segmentations.append(None)
    
    return batch_segmentations


# ============================================================
# 6. AFFICHAGE DES RÉSULTATS EN BATCH
# ============================================================

def display_segmented_images_batch(original_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés, 
    puis les sauvegarde dans un dossier Drive.
    """
    num_images = len(original_image_paths)
    if num_images == 0:
        print("⚠️ Aucune image à afficher.")
        return
    
    # Configuration du chemin
    parent_dir = "content/"
    
    if not os.path.exists(parent_dir) :
        # Création des dossiers de sortie
        base_dir = os.path.join(os.getcwd(), "content")
        os.makedirs(base_dir, exist_ok=True)
        print("📁 Dossier créé :", base_dir)
    else : 
        print(f"Dossier '{parent_dir}' existant.")
        
    # Création (ou vérification) des sous-dossiers
    img_dir = os.path.join(parent_dir, "IMG")
    mask_dir = os.path.join(parent_dir, "Mask")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"📁 Dossiers de sortie prêts :\n   - Images : {img_dir}\n   - Masques : {mask_dir}")
    print(f"📁 Dossiers de sortie prêts dans '{parent_dir}'.")

    # Créer une grille d'affichage (attention si trop d'images)
    #fig, axes = plt.subplots(num_images, 2, figsize=(12, 6 * num_images))
    fig, axes = plt.subplots(
                num_images, 
                2, 
                figsize=(12, 6 * num_images),
                gridspec_kw={'width_ratios': [3, 2]})
                
    if num_images == 1:
        axes = axes.reshape(1, -1)
        
    # Dictionnaire inversé pour la légende
    label_names = {v: k for k, v in CLASS_MAPPING.items()}

    for i, (img_path, seg_mask) in enumerate(zip(original_image_paths, segmentation_masks)):
        # Image originale
        original_img = Image.open(img_path)
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"Image {i+1}: {os.path.basename(img_path)}", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Masque de segmentation
        if seg_mask is not None:
            axes[i, 1].imshow(seg_mask, cmap='tab20')
            axes[i, 1].set_title(f"Segmentation {i+1}", fontsize=12, fontweight='bold')
        else:
            axes[i, 1].text(0.5, 0.5, "Erreur de segmentation",
                            ha='center', va='center', fontsize=14, color='red')
            axes[i, 1].set_title(f"Segmentation {i+1} - ÉCHEC", fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

        # Sauvegarde des images
        try:
        
            # Charger l'image originale
            original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
            # Coloriser le masque
            colored_mask = colorize_mask(seg_mask, custom_colormap)
    
            # Ajouter la légende sur le masque colorisé
            colored_mask_with_legend = add_legend(colored_mask, label_names)
    
            # Superposer masque + image
            overlay = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)
            overlay_with_legend = add_legend(overlay, label_names)
    
            # Afficher les trois versions côte à côte
            concatenated = np.hstack([original_img, colored_mask_with_legend, overlay_with_legend])

            # Affichage des résultats dans Colab
            print(f"Résultat pour la paire {i} :")
            plt.figure(figsize=(18, 6))
            plt.imshow(concatenated)
            plt.axis("off")
            plt.title(f"Image {i+1} – {os.path.basename(img_path)}", fontsize=14, fontweight="bold")
            plt.show()
            
            # Convertir PIL.Image en NumPy (BGR pour OpenCV)
            original_array = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(parent_dir, "IMG", f"image_{i+1}.png"), original_array)

            # Coloriser le masque pour sauvegarde
            mask_colored = cv2.applyColorMap((seg_mask * 10).astype(np.uint8), cv2.COLORMAP_TWILIGHT)
            cv2.imwrite(os.path.join(parent_dir, "Mask", f"mask_{i+1}.png"), mask_colored)
        except Exception as e:
            print(f"⚠️ Erreur de sauvegarde pour {img_path}: {e}")

    plt.tight_layout()
    plt.show()