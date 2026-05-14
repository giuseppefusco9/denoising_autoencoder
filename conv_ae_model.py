import torch
import torch.nn as nn

class ConvAutoencoderDenoise(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ConvAutoencoderDenoise, self).__init__()
        
        # ==========================================
        # ENCODER (Fase di Compressione)
        # ==========================================
        self.encoder = nn.Sequential(
            # Input: [Batch, 3, 256, 256]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Riduce a 128x128
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Riduce a 64x64
            
            # Strato di "Bottleneck"
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        # ==========================================
        # DECODER (Fase di Ricostruzione)
        # ==========================================
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), # Espande a 128x128
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='nearest'), # Espande a 256x256
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==========================================
# TEST DEL MODELLO
# ==========================================
if __name__ == "__main__":
    # Creiamo un tensore "finto" per simulare un batch di 4 immagini 256x256 RGB
    dummy_input = torch.randn(4, 3, 256, 256)
    
    modello = ConvAutoencoderDenoise()
    output = modello(dummy_input)
    
    print(f"Formato Input: {dummy_input.shape}")
    print(f"Formato Output: {output.shape}")
    print("Modello Convolutional Autoencoder (stile Notebook) inizializzato con successo!")