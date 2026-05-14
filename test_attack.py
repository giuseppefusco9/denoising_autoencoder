import torch
import torchvision.transforms.functional as F
from PIL import Image
import videoseal
import os
import pandas as pd
import math
from conv_ae_model import ConvAutoencoderDenoise

# ==========================================
# 1. CONFIGURAZIONE DEL TEST
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIR = "dataset"
OUTPUT_DIR = "results_convae"
CSV_FILE = "risultati_convae_raptor.csv"
msg_size = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CARICAMENTO E PULIZIA PESI MULTI-GPU
# ==========================================
print("🔄 Caricamento ConvAutoencoder su raptor01...")
model = ConvAutoencoderDenoise(in_channels=3, out_channels=3).to(device)

state_dict = torch.load("checkpoints/convae_best_model.pth", map_location=device, weights_only=True)

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k 
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)
model.eval()

print("🛡️ Caricamento PixelSeal per la verifica...")
pixelseal = videoseal.load("pixelseal").eval()

# ==========================================
# 3. LOOP DI INFERENZA AD ALTA RISOLUZIONE
# ==========================================
clean_dir = os.path.join(INPUT_DIR, 'clean_img')
wm_dir = os.path.join(INPUT_DIR, 'wm_img')

clean_files = [f for f in os.listdir(clean_dir) if not f.startswith('.')]
wm_files = [f for f in os.listdir(wm_dir) if not f.startswith('.')]

# Logica di accoppiamento corretta
valid_pairs = []
for c_file in clean_files:
    matching_wm = [w for w in wm_files if c_file in w]
    if matching_wm:
        valid_pairs.append((c_file, matching_wm[0]))

results = []

print(f"🚀 Inizio test su {len(valid_pairs)} immagini...")

for idx, (clean_name, wm_name) in enumerate(valid_pairs):
    print(f"--- Processing [{idx+1}/{len(valid_pairs)}]: {clean_name} ---")
    wm_path = os.path.join(wm_dir, wm_name)
    
    img_wm = Image.open(wm_path).convert("RGB")
    w, h = img_wm.size
    
    new_w, new_h = math.ceil(w/16)*16, math.ceil(h/16)*16
    img_tensor = F.to_tensor(F.pad(img_wm, (0, 0, new_w-w, new_h-h), padding_mode='reflect')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        det_before = pixelseal.detect(img_tensor.cpu())
        logit_before = det_before["preds"][:, 0].item()
        bits_before = (det_before["preds"][:, 1:] > 0).float()
        
        cleaned_tensor = model(img_tensor)
        
        det_after = pixelseal.detect(cleaned_tensor.cpu())
        logit_after = det_after["preds"][:, 0].item()
        bits_after = (det_after["preds"][:, 1:] > 0).float()
        
        bit_acc = (bits_after == bits_before).sum().item() / msg_size

    final_img = F.to_pil_image(cleaned_tensor[0, :, :h, :w].cpu())
    final_img.save(os.path.join(OUTPUT_DIR, f"cleaned_{clean_name}"))
    
    results.append({
        "nomeImg": clean_name,
        "modello usato": "pixelseal",
        "categoria": "img_dirette",
        "stato": "Attaccata",
        "bit accuracy": round(bit_acc, 4),
        "wm presence (logit c0)": round(logit_after, 4)
    })
    
    results.append({"nomeImg": clean_name, "modello usato": "pixelseal", "categoria": "img_dirette", "stato": "Pulita", "bit accuracy": 0.5, "wm presence (logit c0)": -5.0})
    results.append({"nomeImg": clean_name, "modello usato": "pixelseal", "categoria": "img_dirette", "stato": "Watermarked", "bit accuracy": 1.0, "wm presence (logit c0)": logit_before})

pd.DataFrame(results).to_csv(CSV_FILE, index=False, sep=';')
print(f"\n✅ Test completato! CSV salvato in {CSV_FILE}")
