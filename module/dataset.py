from torchvision.datasets import MNIST
from torchvision import transforms



train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = MNIST(root="data/mnist", train=True, download=True, transform=train_transform)
testset = MNIST(root="data/mnist", train=False, download=True, transform=train_transform)
