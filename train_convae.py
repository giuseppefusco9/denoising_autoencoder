import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import os
from loader_dataset import WatermarkDenoisingDataset
from conv_ae_model import ConvAutoencoderDenoise

# ==========================================
# 1. DEFINIZIONE DELLA PERCEPTUAL LOSS
# ==========================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].eval()
        
        for param in vgg.parameters():
            param.requires_grad = False
            
        self.vgg = vgg
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        loss_pixel = self.l1_loss(pred, target)
        
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        loss_perceptual = self.mse_loss(pred_features, target_features)
        
        return loss_pixel + 0.1 * loss_perceptual

# ==========================================
# 2. CONFIGURAZIONE HARDWARE (raptor01)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

print("="*50)
print(f"Dispositivo Principale: {device}")

BATCH_SIZE = 32      
EPOCHS = 60          
LEARNING_RATE = 2e-4 
CROP_SIZE = 256      

# ==========================================
# 3. PREPARAZIONE DATI
# ==========================================
print("Caricamento dataset in corso...")
dataset_completo = WatermarkDenoisingDataset(root_dir="dataset", crop_size=CROP_SIZE)

train_size = int(0.8 * len(dataset_completo))
val_size = len(dataset_completo) - train_size
train_dataset, val_dataset = random_split(dataset_completo, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
print(f"Dati: {train_size} Train | {val_size} Val.")

# ==========================================
# 4. INIZIALIZZAZIONE DEL TUO MODELLO
# ==========================================
model = ConvAutoencoderDenoise(in_channels=3, out_channels=3).to(device)
criterion = VGGPerceptualLoss().to(device)

if num_gpus > 1:
    model = nn.DataParallel(model)
    criterion = nn.DataParallel(criterion)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs("checkpoints", exist_ok=True)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
print("\nInizio Addestramento Convolutional Autoencoder...\n")

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for wm_imgs, clean_imgs in train_loader:
        wm_imgs = wm_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        optimizer.zero_grad()
        
        reconstructed_imgs = model(wm_imgs)
        
        loss = criterion(reconstructed_imgs, clean_imgs)
        if num_gpus > 1:
            loss = loss.mean()
            
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # --- VALIDAZIONE ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for wm_imgs, clean_imgs in val_loader:
            wm_imgs = wm_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            reconstructed_imgs = model(wm_imgs)
            loss = criterion(reconstructed_imgs, clean_imgs)
            if num_gpus > 1:
                loss = loss.mean()
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoca [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        
        state_dict_to_save = model.module.state_dict() if num_gpus > 1 else model.state_dict()
        torch.save(state_dict_to_save, "checkpoints/convae_best_model.pth")
        
        print(" Nuovo record Percettivo! Modello salvato.")

print("\nAddestramento Completato. Pronto per il Test!")