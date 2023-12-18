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
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_path = args.model_path
        self.logger_name = args.logger_name
        
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
    
