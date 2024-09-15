import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_and_standardize_data(path):
    """
    Function to load and standardize data
    """
    df = pd.read_csv(path, sep=",")
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype("float32")
    x_train, x_test = train_test_split(df, test_size=0.3, random_state=42)
    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, scaler


class DataBuilder(Dataset):
    """
    Load dataset and split into train/test
    """

    def __init__(self, path, train=True):
        self.x_train, self.x_test, self.standardizer = load_and_standardize_data(path)
        if train:
            self.x = torch.from_numpy(self.x_train)
            self.len = self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.x_test)
            self.len = self.x.shape[0]
        del self.x_train
        del self.x_test

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


class CustomLoss(nn.Module):
    """
    Custom loss for training
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        """
        Forward pass

        args:
        - x_recon: reconstructed data
        - x: original data
        - mu: mean from the encoder's latent space
        - logvar: log variance from the encoder's latent space
        """
        loss_mse = self.mse_loss(x_recon, x)
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_mse + loss_kld


class Autoencoder(nn.Module):
    """
    Autoencoder model
    """

    def __init__(self, D_in, H=50, H2=12, latent_dim=3):
        # Encoder
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        # Latent vectors
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        """
        Encoder Model

        args:
        - x: input data
        """
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        args:
        - mu: mean from the encoder's latent space
        - logvar: log variance from the encoder's latent space
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        return mu

    def decode(self, z):
        """
        Decoder Model

        args:
        - z: sampled latent vector
        """
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        """
        Forward pass

        args:
        - x: input data
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def generate_fake(mu, logvar, no_samples, scaler, model):
    """
    Generate synthetic data with trained model

    args:
    - mu: mean from the encoder's latent space
    - logvar: log variance from the encoder's latent space
    - no_samples: number of samples to generate
    - scaler: standard scaler
    - model: trained model
    """
    sigma = torch.exp(logvar / 2)
    q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
    z = q.rsample(sample_shape=torch.Size([no_samples]))
    with torch.no_grad():
        pred = model.decode(z).cpu().numpy()
    fake_data = scaler.inverse_transform(pred)
    return fake_data
