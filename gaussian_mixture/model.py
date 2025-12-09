import torch
import torch.nn as nn
import torch.nn.functional as F


#  GENERATOR NETWORK (standard GM-GAN generator)
class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super().__init__()

        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, g_output_dim)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))



#  BASE MIXTURE CLASS
class BaseGaussianMixture(nn.Module):
    def __init__(self, K, latent_dim):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim

    def sample(self, k):
        raise NotImplementedError



#  STATIC: μ fixed, Σ = σI fixed
class StaticGaussianMixture(BaseGaussianMixture):
    def __init__(self, K, latent_dim, c, sigma):
        super().__init__(K, latent_dim)

        # μ fixed, not learnable
        self.register_buffer("mu", torch.FloatTensor(K, latent_dim).uniform_(-c, c))

        # Σ = σI fixed
        Sigma = sigma * torch.eye(latent_dim).unsqueeze(0).repeat(K, 1, 1)
        self.register_buffer("Sigma", Sigma)

    def sample(self, k):
        eps = torch.randn(k.size(0), self.latent_dim, device=self.mu.device)
        A = self.Sigma[k]                      # (batch, d, d)
        return (A @ eps.unsqueeze(-1)).squeeze(-1) + self.mu[k]



#  DYNAMIC FULL COVARIANCE: μ learned, A learned
class DynamicGaussianMixtureFull(BaseGaussianMixture):
    def __init__(self, K, latent_dim, c, sigma):
        super().__init__(K, latent_dim)

        self.mu = nn.Parameter(torch.FloatTensor(K, latent_dim).uniform_(-c, c))

        A_init = sigma * torch.eye(latent_dim).unsqueeze(0).repeat(K, 1, 1)
        self.A = nn.Parameter(A_init)

    def sample(self, k):
        eps = torch.randn(k.size(0), self.latent_dim, device=self.mu.device)
        A_k = self.A[k]                        # (batch, d, d)
        return (A_k @ eps.unsqueeze(-1)).squeeze(-1) + self.mu[k]



#  DYNAMIC DIAGONAL: μ learned, diag(σ) learned
class DynamicGaussianMixtureDiag(BaseGaussianMixture):
    def __init__(self, K, latent_dim, c, sigma):
        super().__init__(K, latent_dim)

        self.mu = nn.Parameter(torch.FloatTensor(K, latent_dim).uniform_(-c, c))
        self.log_sigma = nn.Parameter(torch.log(torch.ones(K, latent_dim) * sigma))

    def sample(self, k):
        eps = torch.randn(k.size(0), self.latent_dim, device=self.mu.device)
        sigma_k = self.log_sigma.exp()[k]
        return sigma_k * eps + self.mu[k]



#  LEARNABLE MEANS, FIXED Σ = σI
class LearnableMeansFixedSigma(BaseGaussianMixture):
    def __init__(self, K, latent_dim, c, sigma):
        super().__init__(K, latent_dim)

        self.mu = nn.Parameter(torch.FloatTensor(K, latent_dim).uniform_(-c, c))

        Sigma = sigma * torch.eye(latent_dim).unsqueeze(0).repeat(K, 1, 1)
        self.register_buffer("Sigma", Sigma)

    def sample(self, k):
        eps = torch.randn(k.size(0), self.latent_dim, device=self.mu.device)
        A = self.Sigma[k]
        return (A @ eps.unsqueeze(-1)).squeeze(-1) + self.mu[k]



#  GMGAN CLASS
class GMGAN(nn.Module):
    def __init__(self, 
                 g_output_dim, 
                 latent_dim, 
                 K, 
                 c, 
                 sigma, 
                 mode="static"):

        super().__init__()

        if mode == "static":
            self.mixture = StaticGaussianMixture(K, latent_dim, c, sigma)

        elif mode == "dynamic_full":
            self.mixture = DynamicGaussianMixtureFull(K, latent_dim, c, sigma)

        elif mode == "dynamic_diag":
            self.mixture = DynamicGaussianMixtureDiag(K, latent_dim, c, sigma)

        elif mode == "perso":
            self.mixture = LearnableMeansFixedSigma(K, latent_dim, c, sigma)

        else:
            raise ValueError(f"Unknown mode {mode}")

        self.g_output_dim = g_output_dim
        self.generator = Generator(g_output_dim)
        self.K = K

    def sample_from_mixture(self, k):
        return self.mixture.sample(k)

    def forward(self, z):
        return self.generator(z)


class DiscriminatorGMGAN(nn.Module):
    def __init__(self, d_input_dim, d_dropout):
        super(DiscriminatorGMGAN, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.dropout = nn.Dropout(p=d_dropout)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        return torch.sigmoid(self.fc4(x))