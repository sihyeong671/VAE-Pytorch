import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch

class Config:
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.mode = args.mode
        self.model_path = args.model_path
        
    def __str__(self) -> str:
        attributes = "---Config---\n"
        for k, v in self.__dict__.items():
            attributes += f"{k}: {v}\n"
        return attributes

# https://bab-dev-study.tistory.com/24
class Logger:
    def __init__(self, logger_name):
        os.makedirs("./log", exist_ok=True)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        log_format = logging.Formatter("%(asctime)s: %(message)s")
        handler = logging.FileHandler(f"./log/{logger_name}.log", mode="w")
        handler.setFormatter(log_format)
        self.logger.addHandler(handler)
        
    def info(self, message):
        self.logger.info(message)
        
def save_images(imgs, img_name):
    
    fig, axs = plt.subplots(ncols=5, nrows=5, squeeze=False, figsize=(12, 12))
    for i, img in enumerate(imgs):
        axs[i//5, i%5].imshow(img)
        axs[i//5, i%5].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    fig.savefig(img_name)
    

# 이미지 보간 시각화
def interpolate(vae, x_1, x_2, n=15):
    
    z_1 = vae.encoder(x_1)
    z_2 = vae.encoder(x_2)
    z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = vae.decoder(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()

    w = 28
    img = np.zeros((w, n*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("interploate_1to0.png")
