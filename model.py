import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTrashDetectorCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleTrashDetectorCNN, self).__init__()
        # Convolutional layers: Input channels = 3 (RGB), Output channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers (Asumsi ukuran citra input adalah 64x64)
        # Setelah 3 kali max pooling (64 -> 32 -> 16 -> 8)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 8 * 8) # Flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = SimpleTrashDetectorCNN()
    print("Arsitektur Model CNN Berhasil Diinisialisasi!")
    print(model)
