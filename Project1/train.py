from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from torchvision.datasets import MNIST
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import torch
from torch import nn
from Dataset import AwA2
from VAE import VAE
from VAE import reparametrize
from tensorboardX import SummaryWriter
# import Dataset
from sklearn.preprocessing import StandardScaler

feature_path = 'RawData/AwA2-features.txt'
label_path = 'RawData/AwA2-labels.txt'
beta = 0.1
save_path = 'models/v-b%.1fVAE-' % beta

epoch_num = 300
batch_size = 64
z_dim = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mode = 'train'


def kl_loss(mu, logvar):
    kld_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld_element.sum()
    return kld


def l1(x, y):
    return torch.sum(torch.abs((x-y)))


def save_z(path):
    data = AwA2()
    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
    net = VAE(z_dim)
    net = net.to(device)
    net.load_state_dict(torch.load(path))
    print('Load model from%s' % path)
    net.eval()
    z_list = np.empty([0, z_dim])
    for f, label in loader:
        print(label)
        f = f.to(device)
        f = f.float()
        recon_f, mu, logvar = net(f)
        z = reparametrize(mu, logvar)
        z = z.detach().cpu().numpy()
        z_list = np.append(z_list, z, axis=0)
        print(z_list.shape)
    np.save('Data/b%.1fVAE-%d.npy' % (beta, z_dim), z_list)


if __name__ == '__main__':
    # labels = np.loadtxt(label_path, dtype=int)
    # features = np.loadtxt(feature_path, delimiter=' ')
    if mode == 'eval':
        save_z(save_path + '%d.pkl' % z_dim)
        exit()
    writer = SummaryWriter()
    data = AwA2()
    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
    net = VAE(z_dim=z_dim)
    net = net.to(device)
    optim = Adam(net.parameters(), )
    recon_loss = nn.MSELoss(reduction='elementwise_mean')
    # recon_loss = l1()
    print('Load data')
    net.train()

    for e in range(epoch_num):
        train_loss = 0
        for f, label in loader:
            f = f.to(device)
            f = f.float()
            recon_f, mu, logvar = net(f)
            loss = (recon_loss(f, recon_f) + beta * kl_loss(mu, logvar)) / f.shape[0]
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
        print('epoch%d, Loss%.4f' % (e, train_loss / len(loader)))
        writer.add_scalar('b0.1VAE_Loss', train_loss / len(loader), e)
    torch.save(net.state_dict(), save_path + '%d.pkl' % z_dim)
    writer.close()





