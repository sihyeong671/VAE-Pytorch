import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from module.dataset import get_dataset
from module.utils import Config, Logger, save_images
from module.model import VAE

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        if self.config.mode == "train":
            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H-%M-%S")
            self.logger = Logger(f"VAE_logger_{now_str}")
    
    def setup(self):
        os.makedirs("models/", exist_ok=True)
        os.makedirs("test/", exist_ok=True)
        if self.config.mode == "train":
            self.logger.info(self.config)
        
        trainset, testset = get_dataset(self.config.dataset_name)
        
        self.train_dataloader = DataLoader(
            dataset=trainset,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        self.test_dataloader = DataLoader(
            dataset=testset,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # 베르누이
        self.bce_loss = nn.BCELoss()
        # 가우시안
        # self.mse_loss = nn.MSELoss()
        self.model = VAE(device=self.config.device).to(self.config.device)
        summary(self.model, input_size=(1, 784))
    
    def _loss_function(self, x_hat, x, mean, logvar):
        # ELBO term
        
        # https://huidea.tistory.com/296
        reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")
        # appendix B(KLD)
        regularization_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + regularization_loss
    
    def train(self):
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.config.lr)
        self.model.train()
        for epoch in range(1, self.config.epochs + 1):
            overall_loss = 0
            for (x, _) in tqdm(self.train_dataloader):
                x = x.view(self.config.batch_size, 784)
                x = x.to(self.config.device)

                optimizer.zero_grad()
                
                x_hat, mean, logvar = self.model(x)
                loss = self._loss_function(x_hat, x, mean, logvar)
                loss.backward()
                
                overall_loss += loss.item()
                optimizer.step()
        
            message = f"Epoch: {epoch}/{self.config.epochs}\t Avg Loss: {overall_loss / (len(self.train_dataloader)*self.config.batch_size):.4f}"
            self.logger.info(message)
        
        torch.save(self.model.state_dict(), self.config.model_path)
                
    def generate(self, img_name):
        self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.config.device))
        self.model.eval()
        with torch.no_grad():
            noise = torch.randn((25, 20)).to(self.config.device)
            gen_imgs = self.model.decoder(noise)
            gen_imgs = gen_imgs.view(-1, 28, 28)
            gen_imgs = gen_imgs.cpu().numpy()
        save_images(gen_imgs, img_name)
    
    def interpolate(self, a=1, b=0, n=15):
        # https://avandekleut.github.io/vae/
        self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.config.device))
        self.model.eval()
        x, y = self.test_dataloader.__iter__().__next__()
        x_1 = x[y == a][1].to(self.config.device).view(-1, 784)
        x_2 = x[y == b][1].to(self.config.device).view(-1, 784)
        with torch.no_grad():
            mean1, logvar1 = self.model.encoder(x_1)
            z_1 = self.model._reparameterization(mean1, logvar1, is_train=False)
            mean2, logvar2 = self.model.encoder(x_2)
            z_2 = self.model._reparameterization(mean2, logvar2, is_train=False)
            z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
            interpolate_list = self.model.decoder(z)
            interpolate_list = interpolate_list.to('cpu').detach().numpy()

            w = 28
            img = np.zeros((w, n*w))
            for i, x_hat in enumerate(interpolate_list):
                img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f"test/interploate_{a}to{b}.png")