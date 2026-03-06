# 🗂 PROJET 2 - AI ENGENEER
# 👗 ModeTrends — Fashion Trend Intelligence

> Segmentation vestimentaire automatique via l'API Hugging Face (SegFormer B3 Clothes)

---

## 📋 Table des matières

- [Présentation](#-présentation)
- [Architecture du projet](#-architecture-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [Classes détectées](#-classes-détectées)
- [Métriques](#-métriques)
- [Structure des sorties](#-structure-des-sorties)
- [Exemples de résultats](#-exemples-de-résultats)

---

## 🎯 Présentation

**ModeTrends** est un pipeline de segmentation vestimentaire qui exploite le modèle [`sayeed99/segformer_b3_clothes`](https://huggingface.co/sayeed99/segformer_b3_clothes) via l'API d'inférence Hugging Face.

Il permet de :
- **Segmenter** automatiquement les vêtements et accessoires sur des photos d'influenceurs
- **Visualiser** les masques de segmentation colorisés avec superposition sur l'image originale
- **Évaluer** la qualité des prédictions via les métriques **IoU** et **mIoU** par classe
- **Sauvegarder** les résultats image par image dans un dossier de sortie structuré

---

## 🗂 Architecture du projet

```
ModeTrends/
│
├── main.py                          # Script principal (pipeline complet)
├── app.py                           # Fonctions utilitaires et segmentation
├── .env                             # Variables d'environnement (token HF)
│
├── jeu_de_donnees/
│   └── top_influenceurs_2024/
│       ├── IMG/                     # Images sources (.jpg / .png)
│       └── GT/                      # Masques Ground Truth (auto-générés)
│
├── content/
│   ├── IMG/                         # Images sauvegardées après traitement
│   ├── Mask/                        # Masques colorisés sauvegardés
│   └── result_N.png                 # Figures complètes par image
│
└── resultats/
    └── segmentation_batch_.png      # Résultat mode image unique
```

---

## ⚙️ Prérequis

- Python **3.8+**
- Un compte [Hugging Face](https://huggingface.co/) avec un token d'accès API

### Dépendances Python

```
requests
Pillow
matplotlib
numpy
tqdm
opencv-python
python-dotenv
```

---

## 🚀 Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/wad974/projet2-ingenieur-ia
cd projet2-ingenieur-ia

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🔑 Configuration

Créez un fichier `.env` à la racine du projet :

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> ⚠️ Ne commitez jamais votre token. Ajoutez `.env` à votre `.gitignore`.

---

## 🖥️ Utilisation

### Paramètres dans `main.py`

| Paramètre | Type | Description |
|-----------|------|-------------|
| `image_dir` | `str` | Chemin vers le dossier d'images à traiter |
| `max_images` | `int` | Nombre maximum d'images à traiter |
| `mode_alone` | `bool` | `True` = une seule image / `False` = batch complet |

### Mode image unique

```python
mode_alone = True
```

```bash
python main.py
```

Affiche côte à côte l'image originale et son masque de segmentation.

### Mode batch

```python
mode_alone = False
max_images = 50
```

```bash
python main.py
```

Traite toutes les images du dossier, affiche les résultats figure par figure avec **la légende des classes et l'IoU associé**, et sauvegarde les sorties dans `content/`.

---

## 🏷️ Classes détectées

Le modèle identifie **18 classes** vestimentaires et corporelles :

| ID | Classe | Couleur |
|----|--------|---------|
| 0 | Background | — |
| 1 | Hat | 🟡 Jaune |
| 2 | Hair | 🟠 Orange |
| 3 | Sunglasses | 🟣 Magenta |
| 4 | Upper-clothes | 🔴 Rouge |
| 5 | Skirt | 🩵 Cyan |
| 6 | Pants | 🟢 Vert |
| 7 | Dress | 🔵 Bleu |
| 8 | Belt | 🟣 Violet |
| 9 | Left-shoe | 🟡 Jaune |
| 10 | Right-shoe | 🟠 Orange foncé |
| 11 | Face | 🟤 Beige |
| 12 | Left-leg | 🟤 Beige |
| 13 | Right-leg | 🟤 Beige |
| 14 | Left-arm | 🟤 Beige |
| 15 | Right-arm | 🟤 Beige |
| 16 | Bag | 🔵 Bleu clair |
| 17 | Scarf | 🩷 Rose |

---

## 📊 Métriques

### IoU (Intersection over Union)

Mesure le chevauchement entre le masque prédit et le masque Ground Truth pour **une classe donnée** :

$$IoU = \frac{|Prediction \cap GT|}{|Prediction \cup GT|}$$

### mIoU (Mean IoU)

Moyenne de l'IoU sur toutes les classes **présentes** dans l'image :

$$mIoU = \frac{1}{N} \sum_{c=1}^{N} IoU_c$$

### Affichage des métriques

Pour chaque image traitée, la figure de résultat affiche :
- Le **mIoU global** dans le titre
- Le **panneau de droite** listant chaque classe avec sa couleur et son IoU individuel (`—` si la classe est absente)

> 💡 Les masques Ground Truth sont auto-générés à la première exécution (`ensure_gt_dataset`) puis peuvent être remplacés par de vrais masques annotés pour une évaluation réelle.

---

## 📁 Structure des sorties

Après exécution en mode batch, les sorties sont organisées comme suit :

```
content/
├── IMG/
│   ├── image_1.png       # Image originale copiée
│   ├── image_2.png
│   └── ...
├── Mask/
│   ├── mask_1.png        # Masque colorisé (colormap TWILIGHT)
│   ├── mask_2.png
│   └── ...
├── result_1.png          # Figure complète : original | masque | overlay | légende IoU
├── result_2.png
└── ...
```

---

## 🖼️ Exemples de résultats

Chaque figure `result_N.png` contient :

```
┌─────────────────────────────────────┬──────────────────┐
│                                     │  Classes & IoU   │
│  [Original] [Masque] [Overlay]      │  Hat        92%  │
│                                     │  Upper-cl.  87%  │
│                                     │  Pants      78%  │
│                                     │  Face        —   │
│                                     │  ...             │
└─────────────────────────────────────┴──────────────────┘
       Titre : Image 1 – photo.jpg  |  mIoU : 83.5%
```

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Ouvrez une *issue* ou soumettez une *pull request*.

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

*ModeTrends _ Projet 2 AI ENGEENER — Powered by [JCWAD](https://www.jcwad.re) - [Hugging Face](https://huggingface.co/) & SegFormer B3*