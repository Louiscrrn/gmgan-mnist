import torch
import os
import yaml
from model import GMGAN, DiscriminatorGMGAN
from torchvision import datasets, transforms
import torch.nn as nn
from trainer import TrainerGMGAN
from datetime import datetime
from dotenv import load_dotenv

def load_MNIST_loader(num_workers: int, batch_discriminator: int, data_path: str, to_download: str, pin_memory: bool) :
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=to_download)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=to_download)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_discriminator,
        shuffle=True,
        num_workers=num_workers,  
        pin_memory=pin_memory  
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_discriminator,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader

if __name__ == '__main__' :
    
    # -- Load config file --
    load_dotenv()
    data_path = os.getenv("DATA_PATH")
    checkpoints_dir = os.getenv("CHECKPOINTS_PATH")
    config_path = os.getenv("CONFIG_PATH")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # -- Create directories --
    cfg_data = config.get("data", {})
    os.makedirs(checkpoints_dir, exist_ok=True)
    to_download = data_path is None or not os.path.exists(data_path)
    cfg_experiment = config.get("experiment", {})
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # -- Environment setup -- 
    cfg_env = config.get("environment", {})
    device = str(cfg_env.get("device"))
    print(f"Using device : {device}")
    gpus = int(cfg_env.get("gpus"))
    if gpus == -1 :
        gpus = torch.cuda.device_count()
        print(f"Using {gpus} GPUs.")

    # -- Data Pipeline --
    cfg_train = config.get('training', {})
    num_workers = int(cfg_data['num_workers'])
    batch_discriminator = int(cfg_train['batch_discriminator'])
    batch_generator = int(cfg_train['batch_generator'])
    pin_memory = False if device == 'mps' else bool(cfg_data['pin_memory']) 
    print('Dataset loading...')
    train_loader, test_loader = load_MNIST_loader(num_workers, batch_discriminator, data_path, to_download, pin_memory)
    print('Dataset loaded.')

    # -- Model setup --
    print('Model loading...')
    cfg_model = config.get("model", {})
    mnist_dim = int(cfg_model.get("mnist_dim"))
    latent_dim = int(cfg_model.get('latent_dim'))
    d_dropout = float(cfg_model.get('d_dropout'))
    mode = str(cfg_model.get('mode'))
    K = int(cfg_model.get('K'))
    sigma = float(cfg_model.get('sigma'))
    c = float(cfg_model.get('c'))
    G = GMGAN(
        g_output_dim=mnist_dim,
        latent_dim=latent_dim,
        K=K,
        c=c,
        sigma=sigma,
        mode=mode
        ).to(device)
    D = DiscriminatorGMGAN(mnist_dim, d_dropout=d_dropout).to(device)
    if gpus > 1:
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
    print('Model loaded.')

    # -- Training Settings --
    cfg_train = config.get("training", {})
    lr = float(cfg_train.get("learning_rate"))
    d_noise_std = float(cfg_train.get('d_noise_std'))
    n_epochs= int(cfg_train.get("epochs"))
    optimizer = str(cfg_train.get("optimizer"))
    criterion = nn.BCELoss()
    
    # -- Running train --
    trainerGMGAN = TrainerGMGAN(
        generator=G,
        discriminator=D,
        train_loader=train_loader,
        criterion=criterion,
        batch_generator=batch_generator,
        d_noise_std=d_noise_std,
        lr=lr,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device)
    df_historic = trainerGMGAN.fit()
    
    # -- Savings --
    sigma_str = f"{sigma:.3g}"
    c_str = f"{c:.3g}"
    ddrop_str = f"{d_dropout:.3g}"
    exp_name = (
        f"{cfg_experiment['name']}"
        f"_ld{latent_dim}"
        f"_K{K}"
        f"_sig{sigma_str}"
        f"_c{c_str}"
        f"_dd{ddrop_str}"
        f"_lr{lr:.0e}"
        f"_bd{batch_discriminator}"
        f"_bg{batch_generator}"
        f"_{timestamp}_"
    )
    subfolder = "local/" if bool(cfg_env.get("local")) else "slurm/"
    target_dir = checkpoints_dir + subfolder + exp_name + "/"
    os.makedirs(target_dir, exist_ok=True)
    df_historic.to_csv(target_dir + 'historic.csv')
    trainerGMGAN.save_models(target_dir)
