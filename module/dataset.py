from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms



train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])


def get_dataset(dataset_name):
    if dataset_name == "mnist":
        trainset = MNIST(root="data/mnist", train=True, download=True, transform=train_transform)
        testset = MNIST(root="data/mnist", train=False, download=True, transform=train_transform)
    elif dataset_name == "cifar10":
        trainset = CIFAR10(root="data/cifar10", train=True, download=True, transform=train_transform)
        testset = CIFAR10(root="data/cifar10", train=False, download=True, transform=train_transform)
    return trainset, testset
    
