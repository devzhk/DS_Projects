import torch
from torch import nn
from torch.autograd import Variable


# latent_dim = 128


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + eps * std


class VAE(nn.Module):
    def __init__(self, z_dim=128):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * self.z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
        )
        # self.weight_init()

    def forward(self, x):
        distr = self.encoder(x)
        mu = distr[:, :self.z_dim]
        logvar = distr[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

