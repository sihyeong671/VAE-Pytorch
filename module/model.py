import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 400)
        self.layer2 = nn.Linear(400, 200)
        self.layer_mean = nn.Linear(200, 20)
        self.layer_logvar = nn.Linear(200, 20)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        mean = self.layer_mean(x)
        # var는 항상 양수여야하는데 linear를 거친 값이 음수가 될 수 있기에 logvar로 봄
        # 정확한 표현은 ln var
        logvar = self.layer_logvar(x)
        return mean, logvar
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(20, 200)
        self.layer2 = nn.Linear(200, 400)
        self.layer3 = nn.Linear(400, 784)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = F.sigmoid(x)
        return x

# class EncoderCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(784, 400)
#         self.layer2 = nn.Linear(400, 200)
#         self.layer_mean = nn.Linear(200, 20)
#         self.layer_logvar = nn.Linear(200, 20)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#         x = self.relu(x)
#         mean = self.layer_mean(x)
#         # var는 항상 양수여야하는데 linear를 거친 값이 음수가 될 수 있기에 logvar로 봄
#         # 정확한 표현은 ln var
#         logvar = self.layer_logvar(x)
#         return mean, logvar
    

# class DecoderCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Linear(20, 200)
#         self.layer2 = nn.Linear(200, 400)
#         self.layer3 = nn.Linear(400, 784)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#         x = self.relu(x)
#         x = self.layer3(x)
#         x = F.sigmoid(x)
#         return x



class VAE(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def _reparameterization(self, mean, logvar, is_train=True):
        # https://ai.stackexchange.com/questions/17873/why-is-exp-used-in-encoder-of-vae-instead-of-using-the-value-of-standard-deviati
        # https://stackoverflow.com/questions/75415564/why-vae-in-keras-has-an-exponential
        epsilon = torch.randn_like(logvar).to(self.device)
        std = torch.exp(0.5 * logvar)
        if is_train:
            z = mean + std*epsilon
        else:
            z = mean + std
        return z
    
    def forward(self, x):
        mean, std = self.encoder(x)
        z = self._reparameterization(mean, std)
        output = self.decoder(z)
        return output, mean, std
        