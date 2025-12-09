import torch 
import torchvision
import os
import argparse


from model import Generator
from utils import load_model, sample_from_mixture

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()



    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Metal)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784
    path ='/Users/louiscarron/Documents/SCOLARITE/4A/cours/DSLab/Assignment_2/repo/assignment2-2025-dsl/gaussian_mixture/checkpoints/slurm/gmm_ld100_K10_sig0.1_c0.1_dd0.3_lr2e-04_bd64_bg128_2025-11-15_00-30_/'
    model = Generator(g_output_dim=mnist_dim).to(device)
    model = load_model(model, path, device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    print('Model loaded.')

    mixture = torch.load(path + "mu_clusters.pth", map_location=device)
    mu_clusters = mixture["mu_clusters"].to(device) 

    print('Start Generating')

    for sigma in [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2] :
        os.makedirs(path + f'samples_{sigma}', exist_ok=True)
        n_samples = 0
        with torch.no_grad():
            while n_samples<10000:
                #z = torch.randn(args.batch_size, 100).to(device)
                z = sample_from_mixture(mu_clusters, sigma=sigma, batch_size=args.batch_size, device=device)
                x = model(z)
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples<10000:
                        torchvision.utils.save_image(x[k:k+1], os.path.join(path + f'samples_{sigma}/' , f'{n_samples}.png'))         
                        n_samples += 1
        print(f"End {sigma}")


    
