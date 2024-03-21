import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from your_dataset_module import YourDatasetClass
from Models import UNet  # UNet modelini içeren bir modül

# GPU kullanımı için kontrol
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

