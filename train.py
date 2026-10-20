import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleTrashDetectorCNN

# Fungsi generator data dummy untuk keperluan simulasi training
def get_dummy_data(batch_size=16):
    # Membuat tensor acak untuk simulasi 16 gambar berukuran 3x64x64
    inputs = torch.randn(batch_size, 3, 64, 64) 
    # Label acak (0: Plastik, 1: Organik, 2: Kertas)
    labels = torch.randint(0, 3, (batch_size,))
    return inputs, labels

def train():
    print("Memulai proses training model Deteksi Polusi Lingkungan...")
    
    # Inisialisasi model, loss function, dan optimizer
    model = SimpleTrashDetectorCNN(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        inputs, labels = get_dummy_data()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass, backward pass, dan optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}")

    print("Proses training simulasi selesai!")
    
    # Menyimpan bobot model
    torch.save(model.state_dict(), 'trash_detector_weights.pth')
    print("Bobot model telah disimpan di 'trash_detector_weights.pth'")

if __name__ == '__main__':
    train()
