from model import GMGAN, DiscriminatorGMGAN
import torch.optim as optim
import torch
import os
from tqdm import tqdm
import pandas as pd

class TrainerGMGAN :

    def __init__(self, generator, discriminator, train_loader, criterion, batch_generator: int, d_noise_std:float, lr:float, n_epochs:int, device:str, optimizer:str ='Adam'):
        
        self.n_epochs = n_epochs
        self.lr = lr
        self.d_noise_std = d_noise_std

        self.train_loader = train_loader
        self.batch_generator = batch_generator
        
        self.criterion = criterion
        self.device = device

        self.Generator = generator.to(device)
        self.Discriminator = discriminator.to(device)
    
        self.G_optimizer = optim.Adam(self.Generator.parameters(), lr=self.lr) if optimizer == 'Adam' else None
        self.D_optimizer = optim.Adam(self.Discriminator.parameters(), lr=self.lr) if optimizer == 'Adam' else None

    
    def fit(self,):

        historic = []
        for epoch in range(1, self.n_epochs + 1):

            loop_d = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch [{epoch}/{self.n_epochs}]",
                leave=False
            )

            # -- Train the discriminator -- 
            d_losses = []
            g_losses = []
            for _, (real_batch, _) in loop_d:
                
                x_real = real_batch.to(self.device)
                x_real = x_real.view(-1, self.Generator.g_output_dim).to(self.device)

                # Update Discriminator on the real_batch of size batch_discriminator
                d_loss = self.D_train(x_real)
                
                # -- Train the generator -- 
                g_loss = self.G_train()

                # Metric savings
                d_losses.append(d_loss) 
                g_losses.append(g_loss)


            avg_d_loss = torch.mean(torch.tensor(d_losses))
            avg_g_loss = torch.mean(torch.tensor(g_losses))
            print(
                    f"Epoch [{epoch}/{self.n_epochs}] "
                    f"--> D_loss: {avg_d_loss:.4f} | "
                    f"G_loss: {avg_g_loss:.4f}"
            )

            historic.append({
                'epoch': epoch,
                'd_loss' : float(avg_d_loss),
                'g_loss' : float(avg_g_loss)
            })
            
        return pd.DataFrame(historic)


    def D_train(self, x_reals):
        #=======================Train the discriminator=======================#
        self.D_optimizer.zero_grad()

        # Real samples
        x_reals = x_reals.to(self.device)
        x_reals = x_reals + self.d_noise_std * torch.randn_like(x_reals)
        y_real = torch.ones(x_reals.shape[0], 1, device=self.device)
        
        # Generate reals
        D_reals_out = self.Discriminator(x_reals)
        
        # Draw fake samples
        k_fakes = torch.randint(0, self.Generator.K, (x_reals.shape[0],), device=self.device)
        z = self.Generator.sample_from_mixture(k_fakes)
        x_fakes = self.Generator(z).detach()
        y_fakes = torch.zeros(x_reals.shape[0], 1, device=self.device)

        # Generate fakes
        D_fakes_out = self.Discriminator(x_fakes)

        # Evaluate & Backpropagate
        #D_real_loss = self.criterion(D_reals_out, y_real)
        #D_fake_loss = self.criterion(D_fakes_out, y_fakes)
        #D_loss = D_real_loss + D_fake_loss
        D_loss = self.compute_D_loss(D_reals_out, D_fakes_out)
        
        D_loss.backward()
        self.D_optimizer.step()

        return D_loss.data.item()

    def compute_D_loss(self, reals, fakes) :
        L_real = -torch.log( reals + 1e-8)
        L_fake = -torch.log( 1 - fakes + 1e-8)
        loss = (1/2) * ((L_real + L_fake).mean())
        return loss

    def G_train(self):
        #=======================Train the generator=======================#
        self.G_optimizer.zero_grad()

        # Draw fake samples
        k_fakes = torch.randint(0, self.Generator.K, (self.batch_generator,), device=self.device)
        z = self.Generator.sample_from_mixture(k_fakes)

        # Generate images and labels
        x_fakes = self.Generator(z).to(self.device)

        # Discriminate
        D_output = self.Discriminator(x_fakes)

        # evaluate
        G_loss = self.compute_G_loss(D_output)

        # Backpropagate
        G_loss.backward()
        self.G_optimizer.step()
        
        return G_loss.data.item()
    
    def compute_G_loss(self, fakes):
        L_fakes = -torch.log(fakes + 1e-8)
        loss = L_fakes.mean()
        return loss


    def save_models(self, folder: str):
        save_gmgan_weights(self.Generator, folder)
        torch.save(self.Discriminator.state_dict(), os.path.join(folder, "D.pth"))

    def eval(self, test_loader):

        return 0



def save_gmgan_weights(model: GMGAN, target_dir: str):
    """
    Sauvegarde propre des poids GMGAN :
    - generator.pth
    - mixture_mu.pth (si présent)
    - mixture_sigma.pth ou mixture_log_sigma.pth ou mixture_A.pth selon l'architecture
    """

    os.makedirs(target_dir, exist_ok=True)

    state = model.state_dict()

    generator_state = {k.replace("generator.", ""): v 
                       for k, v in state.items() 
                       if k.startswith("generator.")}
    torch.save(generator_state, os.path.join(target_dir, "G.pth"))

    mix = model.mixture

    if hasattr(mix, "mu"):
        torch.save(mix.mu.detach().cpu(), os.path.join(target_dir, "mixture_mu.pth"))

    if hasattr(mix, "log_sigma"):
        torch.save(mix.log_sigma.detach().cpu(), os.path.join(target_dir, "mixture_log_sigma.pth"))

    if hasattr(mix, "A"):
        torch.save(mix.A.detach().cpu(), os.path.join(target_dir, "mixture_A.pth"))

    if hasattr(mix, "Sigma") and isinstance(mix.Sigma, torch.Tensor):
        torch.save(mix.Sigma.detach().cpu(), os.path.join(target_dir, "mixture_Sigma.pth"))

    print(f"GMGAN weights saved in {target_dir}")
