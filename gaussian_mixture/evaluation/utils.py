import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def save_generated_grid(samples_dir, save_path, n_samples=25):
    """
    Charge les PNG de `samples_dir`, construit une grille d'affichage,
    puis enregistre l'image dans `save_path` (sans affichage).
    """

    # --- Charger les fichiers PNG ---
    files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
    if len(files) == 0:
        raise ValueError(f"Aucune image .png trouvée dans : {samples_dir}")

    # Tri numérique
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    files = files[:n_samples]

    # Transformation (compatible MNIST)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    # Charger les images
    images = []
    for f in files:
        img = Image.open(os.path.join(samples_dir, f))
        img = transform(img)[0]
        images.append(img)

    # --- Construire la grille ---
    cols = min(5, n_samples)
    rows = math.ceil(n_samples / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            ax.imshow(images[i], cmap='gray')
        ax.axis('off')

    plt.tight_layout()

    # --- Sauvegarde ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)   # important pour éviter les fuites mémoire
