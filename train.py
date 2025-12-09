import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from utils import D_train, G_train, save_models
import pandas as pd
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN on MNIST.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches for SGD.")
    parser.add_argument("--gpus", type=int, default=-1, help="Number of GPUs to use (-1 for all available).")
    args = parser.parse_args()

    to_download=False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        print(f"Using device: CUDA")
        # Use all available GPUs if args.gpus is -1
        if args.gpus == -1:
            args.gpus = torch.cuda.device_count()
            print(f"Using {args.gpus} GPUs.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        print(f"Using device: CPU")
        

    

    # Create directories
    os.makedirs('checkpoints/others/', exist_ok=True)
    data_path = os.getenv('DATA')
    if data_path is None:
        data_path = "data"
        to_download = True
    # Data Pipeline
    print('Dataset loading...')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Use multiple workers for data loading
        pin_memory=False  # Faster data transfer to GPU
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    print('Dataset loaded.')

    # Model setup
    print('Model loading...')
    mnist_dim = 784
    G = Generator(g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # Wrap models in DataParallel if multiple GPUs are available
    if args.gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')

    # Loss and optimizers
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)



    print('Start training:')
    n_epoch = args.epochs
    historic = []
    for epoch in range(1, n_epoch + 1):

        loop_d = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch [{epoch}/{n_epoch}] - Discriminator",
                leave=False
            )

        d_losses = []
        g_losses = []
        for batch_idx, (x, _) in loop_d:
            x = x.view(-1, mnist_dim).to(device)
            d_loss = D_train(x, G, D, D_optimizer, criterion, device)
            g_loss = G_train(x, G, D, G_optimizer, criterion, device)

            d_losses.append(d_loss)
            g_losses.append(g_loss)
        
        print(f"Epoch : {epoch} | d_loss : {np.mean(d_losses)} | g_loss : {np.mean(g_losses)}")
        historic.append({
            "epoch" : epoch,
            "d_loss" : np.mean(d_losses),
            "g_loss" : np.mean(g_losses)
        })
        

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints/others/')

    df = pd.DataFrame(historic)
    df.to_csv('checkpoints/others/historic.csv')

    print('Training done.')
