import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from utils import save_generated_grid
import scipy.linalg


def load_mnist_local(mnist_path, batch_size=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_set = datasets.MNIST(
        root=mnist_path,
        train=False,
        download=False,  # <<< EMPÊCHE LE TÉLÉCHARGEMENT
        transform=transform
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    images, labels = next(iter(test_loader))

    return images

def load_fake_images(samples_dir):
    """
    Charge des images générées depuis un dossier `samples_dir`
    contenant des fichiers nommés 0.png, 1.png, ..., N.png.

    Retourne : Tensor [N, 1, 28, 28] normalisé comme MNIST.
    """
    # Récupère uniquement les .png
    files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]

    # Tri numérique : "10.png" vient après "9.png"
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    imgs = []
    for f in files:
        path = os.path.join(samples_dir, f)
        img = Image.open(path)
        img = transform(img)
        imgs.append(img)

    return torch.stack(imgs) 

def load_classifier(ckpt_path, device="cpu"):
    from get_classifier import MNIST_CNN
    model = MNIST_CNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

def extract_features(model, images, device="cpu"):
    """
    images: Tensor [N, 1, 28, 28]
    returns: numpy array [N, D]
    """
    with torch.no_grad():
        # On coupe juste les deux dernières couches (les FC)
        features = model.net[:-2](images.to(device))
        features = features.view(features.size(0), -1)  # flatten
    return features.cpu().numpy()

def pairwise_distances(A, B):
    """
    A: [N1, D]
    B: [N2, D]
    returns: [N1, N2]
    """
    A2 = np.sum(A*A, axis=1, keepdims=True)
    B2 = np.sum(B*B, axis=1, keepdims=True).T
    AB = A @ B.T
    return np.maximum(A2 - 2*AB + B2, 0.0)

def compute_kNN_radii(ref_features, k=3):
    """
    ref_features: [N, D]
    Return: radii: [N] distance au kNN pour chaque point (k ≈ 3 recommandé)
    """
    D = pairwise_distances(ref_features, ref_features)
    # Remove diagonal so nearest neighbor is not itself
    np.fill_diagonal(D, np.inf)
    knn_radii = np.partition(D, k, axis=1)[:, k]
    return knn_radii

def compute_precision(ref_feats, gen_feats, k=3):
    """
    ref_feats: numpy [Nr, D]
    gen_feats: numpy [Ng, D]
    """
    radii = compute_kNN_radii(ref_feats, k=k)  # [Nr]
    D = pairwise_distances(gen_feats, ref_feats)  # [Ng, Nr]

    # Un sample est dans le manifold si ∃ un ref tel que dist ≤ rayon_ref
    inside = (D <= radii[None, :]).any(axis=1)
    precision = inside.mean()
    return precision

def compute_recall(ref_feats, gen_feats, k=3):
    radii = compute_kNN_radii(gen_feats, k=k)  # rayon basé sur les générés
    D = pairwise_distances(ref_feats, gen_feats)  # [Nr, Ng]

    inside = (D <= radii[None, :]).any(axis=1)
    recall = inside.mean()
    return recall

def compute_fid(ref_feats, gen_feats):
    """
    FID based on Wasserstein-2
    """

    mu_r = np.mean(ref_feats, axis=0)
    mu_g = np.mean(gen_feats, axis=0)

    sigma_r = np.cov(ref_feats, rowvar=False)
    sigma_g = np.cov(gen_feats, rowvar=False)

    # (mu_r − mu_g)^2
    diff = mu_r - mu_g
    diff_sq = diff @ diff

    # sqrtm(Sigma_r * Sigma_g)
    cov_sqrt, _ = scipy.linalg.sqrtm(sigma_r @ sigma_g, disp=False)

    # numerical cleanup (imaginary-value noise)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = diff_sq + np.trace(sigma_r + sigma_g - 2*cov_sqrt)
    return float(fid)

if __name__ == "__main__":
    
    classifier_path='checkpoints/mnist_classifier/2025-11-15_12-03/classifier.pth'
    device="mps"
    k=3
    cnn_classifier = load_classifier(classifier_path, device=device)

    mnist_path = "../../data/"
    real_images = load_mnist_local(mnist_path, batch_size=10000)
    real_images = real_images.to(device)

    experience_path = f"/Users/louiscarron/Documents/SCOLARITE/4A/cours/DSLab/Assignment_2/repo/assignment2-2025-dsl/gaussian_mixture/checkpoints/local/"

    samples_dir = experience_path + f"samples/"
    fake_images = load_fake_images(samples_dir)
    fake_images = fake_images.to(device)

    ref_feats = extract_features(cnn_classifier, real_images, device=device)
    gen_feats = extract_features(cnn_classifier, fake_images, device=device)

    P = compute_precision(ref_feats, gen_feats, k=k)
    R = compute_recall(ref_feats, gen_feats, k=k)
    FID = compute_fid(ref_feats, gen_feats)
    
    print(f"Precision={P:.4f} | Recall={R:.4f} | FID={FID:.2f}")
    
    recalls = []
    precisions = []
    fids = []
    sigmas = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

    experience_path = f"/Users/louiscarron/Documents/SCOLARITE/4A/cours/DSLab/Assignment_2/repo/assignment2-2025-dsl/gaussian_mixture/checkpoints/slurm/gmm_ld100_K10_sig0.1_c0.1_dd0.3_lr2e-04_bd64_bg128_2025-11-15_00-30_/"
    asset_dir = experience_path + "assets/"
    os.makedirs(asset_dir, exist_ok=True)
    for sigma in sigmas :
        samples_dir = experience_path + f"samples_{sigma}/"
        fake_images = load_fake_images(samples_dir)
        fake_images = fake_images.to(device)

        ref_feats = extract_features(cnn_classifier, real_images, device=device)
        gen_feats = extract_features(cnn_classifier, fake_images, device=device)

        P = compute_precision(ref_feats, gen_feats, k=k)
        R = compute_recall(ref_feats, gen_feats, k=k)
        FID = compute_fid(ref_feats, gen_feats)

        precisions.append(P)
        recalls.append(R)
        fids.append(FID)

        print(f"[σ={sigma}] Precision={P:.4f} | Recall={R:.4f} | FID={FID:.2f}")

        save_path = os.path.join(asset_dir, f"samples_grid_sigma_{sigma}.png")
        save_generated_grid(samples_dir, save_path, n_samples=5)

    plt.figure(figsize=(7,5))
    plt.plot(sigmas, precisions, marker='o', label='Precision')
    plt.plot(sigmas, recalls, marker='s', label='Recall')
    plt.xlabel("Sigma")
    plt.ylabel("Metric value")
    plt.title("Precision & Recall en fonction de σ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(sigmas, fids, marker='^')
    plt.xlabel("Sigma")
    plt.ylabel("FID")
    plt.title("FID en fonction de σ")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
