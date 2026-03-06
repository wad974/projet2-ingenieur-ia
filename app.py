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

#function metrique 
def compute_iou(pred_mask, true_mask):
    """
    Calcule l'Intersection over Union entre deux masques.
    Retourne le score en pourcentage.
    """
    
    # Vérification dimensions
    if pred_mask.shape != true_mask.shape:
        raise ValueError("Les masques doivent avoir la même taille")
    
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    
    if np.sum(union) == 0:
        return 100.0
    
    iou_score = np.sum(intersection) / np.sum(union)
    
    return iou_score * 100

def compute_mean_iou(pred_mask, true_mask, num_classes=18):
    """
    Calcule le Mean IoU pour toutes les classes.
    """
    
    ious = []
    
    for class_id in range(num_classes):
        
        pred_class = pred_mask == class_id
        true_class = true_mask == class_id
        
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()
        
        if union == 0:
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    if len(ious) == 0:
        return 0
    
    return np.mean(ious) * 100

def compute_per_class_iou(pred_mask, true_mask, num_classes=18):
    """
    Calcule l'IoU par classe entre le masque prédit et le masque GT.
    Retourne un dict {class_id: iou_percent} uniquement pour les classes présentes.
    """
    per_class_iou = {}
    for class_id in range(num_classes):
        pred_class = pred_mask == class_id
        true_class = true_mask == class_id
        intersection = np.logical_and(pred_class, true_class).sum()
        union = np.logical_or(pred_class, true_class).sum()
        if union == 0:
            continue
        per_class_iou[class_id] = (intersection / union) * 100
    return per_class_iou

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
            
            # METRIQU 
            base = os.path.basename(img_path)
            name = os.path.splitext(base)[0]

            true_mask_path = f"jeu_de_donnees/top_influenceurs_2024/GT/{name}.png"

            true_mask = cv2.imread(true_mask_path, 0)

            if true_mask is None:
                print("GT mask introuvable:", true_mask_path)
                miou = 0

            else:

                # DEBUG
                print("Pred classes:", np.unique(seg_mask))
                print("GT classes:", np.unique(true_mask))

                # normalisation possible
                true_mask[true_mask == 255] = 4

                # resize si nécessaire
                if seg_mask.shape != true_mask.shape:
                    true_mask = cv2.resize(
                        true_mask,
                        (seg_mask.shape[1], seg_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                miou = compute_mean_iou(seg_mask, true_mask)
                
                print(f"Image {i+1} - mIoU : {miou:.2f}%")
            # Affichage des résultats dans Colab
            ''''
            print(f"Résultat pour la paire {i} :")
            plt.figure(figsize=(18, 6))
            plt.imshow(concatenated)
            plt.axis("off")
            plt.title(f"Image {i+1} – {os.path.basename(img_path)}", fontsize=14, fontweight="bold")
            plt.show()
            plt.savefig(f"content/result_{i+1}.png", bbox_inches="tight")
            plt.close()'''
            
            # --- Calcul IoU par classe ---
            if true_mask is not None:
                per_class_iou = compute_per_class_iou(seg_mask, true_mask)
            else:
                per_class_iou = {}

            # --- Construction de la figure avec panneau légende ---
            fig2, axes2 = plt.subplots(
                1, 2,
                figsize=(22, 7),
                gridspec_kw={'width_ratios': [3, 1]}   # image large, légende étroite
            )
            fig2.suptitle(
                f"Image {i+1} – {os.path.basename(img_path)}   |   mIoU : {miou:.2f}%",
                fontsize=14, fontweight="bold"
            )

            # Panneau gauche : image concaténée (original + masque coloré + overlay)
            axes2[0].imshow(concatenated)
            axes2[0].axis("off")

            # Panneau droit : légende classe + couleur + IoU
            ax_leg = axes2[1]
            ax_leg.set_xlim(0, 1)
            ax_leg.set_ylim(0, len(CLASS_MAPPING))
            ax_leg.axis("off")
            ax_leg.set_title("Classes & IoU", fontsize=11, fontweight="bold")

            for row_idx, (class_name, class_id) in enumerate(
                    sorted(CLASS_MAPPING.items(), key=lambda x: x[1])):

                y_pos = len(CLASS_MAPPING) - 1 - row_idx   # du haut vers le bas

                # Carré de couleur
                bgr = custom_colormap.get(class_id, (200, 200, 200))
                rgb_norm = (bgr[2]/255, bgr[1]/255, bgr[0]/255)   # BGR → RGB normalisé [0-1]
                rect = plt.Rectangle((0.01, y_pos + 0.1), 0.12, 0.7,
                                    color=rgb_norm, transform=ax_leg.transData)
                ax_leg.add_patch(rect)

                # Score IoU (ou "—" si classe absente des deux masques)
                iou_val = per_class_iou.get(class_id)
                iou_str = f"{iou_val:.1f}%" if iou_val is not None else "—"

                # Texte : nom de classe + IoU
                ax_leg.text(
                    0.17, y_pos + 0.45,
                    f"{class_name:<15} {iou_str:>7}",
                    va="center", fontsize=8,
                    fontfamily="monospace",
                    color="black"
                )

            plt.tight_layout()
            plt.savefig(f"content/result_{i+1}.png", bbox_inches="tight")
            plt.show()
            plt.close(fig2)
            
            
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
    
    
import os
import cv2
import numpy as np

def ensure_gt_dataset(img_paths, segmentation_masks):
    """
    Crée automatiquement le dossier GT et génère des masques GT
    à partir des prédictions (pseudo ground truth).
    """

    gt_dir = "jeu_de_donnees/top_influenceurs_2024/GT"

    # Création du dossier GT
    os.makedirs(gt_dir, exist_ok=True)

    print(f"📁 Dossier GT prêt : {gt_dir}")

    for img_path, mask in zip(img_paths, segmentation_masks):

        if mask is None:
            continue

        # récupérer le nom du fichier
        base = os.path.basename(img_path)
        name = os.path.splitext(base)[0]

        gt_path = os.path.join(gt_dir, f"{name}.png")

        # sauvegarde du masque
        cv2.imwrite(gt_path, mask)

        print(f"✅ GT créé : {gt_path}")