import torch 
import torchvision
import os
import argparse

from model import GMGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=32,
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
    latent_dim = 100
    K = 10
    c = 0.1
    sigma = 0.1

    model = GMGAN(
        g_output_dim=mnist_dim, 
        latent_dim=latent_dim,  
        K=K, 
        c=c, 
        sigma=sigma,
        mode="perso"
        ).to(device)
    exp_path = f"/Users/louiscarron/Documents/SCOLARITE/4A/cours/DSLab/Assignment_2/repo/assignment2-2025-dsl/gaussian_mixture/checkpoints/slurm/gmm_ld100_K10_sig0.1_c0.1_dd0.3_lr2e-04_bd64_bg128_2025-11-15_00-30_/"
    checkpoint_path = f"{exp_path}/G_GMGAN.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    os.makedirs(exp_path + '/samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = model.sample_from_mixture(torch.randint(0, K, (args.batch_size,)))
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join(exp_path + '/samples', f'{n_samples}.png'))         
                    n_samples += 1


    
