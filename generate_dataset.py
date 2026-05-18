import os
import random
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from PIL import Image
import videoseal

# ==========================================
# CONFIGURAZIONE PATH E PARAMETRI
# ==========================================
SOURCE_DIR = "dataset/clean_img"  # Cartella originale con le immagini
OUT_ROOT = "dataset_minSize"      # Cartella di destinazione principale
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fissiamo il seed per garantire che lo split sia sempre identico
# in caso di riavvii dello script
random.seed(42)

def main():
    # ==========================================
    # 1. Trovare la dimensione minima globale
    # ==========================================
    print("🔍 Ricerca della dimensione minima nel dataset...")
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(files) == 0:
        print("Errore: Nessuna immagine trovata nella cartella di origine.")
        return
        
    min_size = float('inf')
    for f in files:
        img_path = os.path.join(SOURCE_DIR, f)
        with Image.open(img_path) as img:
            w, h = img.size
            min_size = min(min_size, w, h)
            
    print(f"✅ Dimensione minima trovata: {min_size}x{min_size}")
    random.shuffle(files)
    
    # Dividiamo le immagini base prima di ritagliarle:
    # 50 img base * 4 ritagli = 200 img (Train)
    # 25 img base * 4 ritagli = 100 img (Val)
    # 25 img base * 4 ritagli = 100 img (Test)
    splits = {
        "train": files[:50],
        "val": files[50:75],
        "test": files[75:]
    }

    # ==========================================
    # 3. Caricamento di PixelSeal
    # ==========================================
    print(f"🛡️ Caricamento modello PixelSeal su {DEVICE}...")
    try:
        pixelseal = videoseal.load("pixelseal").to(DEVICE).eval()
    except Exception as e:
        print(f"Errore nel caricamento di videoseal: {e}")
        print("Assicurati di lanciare lo script nell'ambiente corretto (unet_1080ti).")
        return
    
    # ==========================================
    # 4. Generazione Dataset e Embedding
    # ==========================================
    for split_name, split_files in splits.items():
        print(f"\n⚙️ Generazione set: {split_name.upper()} ({len(split_files)} immagini base -> {len(split_files)*4} ritagli)")
        
        # Creiamo le sottocartelle per Train/Val/Test
        clean_out_dir = os.path.join(OUT_ROOT, split_name, "clean_img")
        wm_out_dir = os.path.join(OUT_ROOT, split_name, "wm_img")
        os.makedirs(clean_out_dir, exist_ok=True)
        os.makedirs(wm_out_dir, exist_ok=True)
        
        for f in split_files:
            img_path = os.path.join(SOURCE_DIR, f)
            img = Image.open(img_path).convert("RGB")
            base_name, ext = os.path.splitext(f)
            
            for i in range(4):
                # A. Ritaglio Casuale (Random Crop)
                top, left, h, w = T.RandomCrop.get_params(img, output_size=(min_size, min_size))
                cropped_img = TF.crop(img, top, left, h, w)
                
                # B. Salvataggio Immagine Pulita
                crop_filename = f"{base_name}_crop{i}{ext}"
                clean_save_path = os.path.join(clean_out_dir, crop_filename)
                cropped_img.save(clean_save_path)
                
                # C. Applicazione PixelSeal
                # Trasformiamo l'immagine in tensore [1, 3, H, W] per la rete
                img_tensor = TF.to_tensor(cropped_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    # Nota: l'API di Meta per l'inserimento solitamente usa .embed()
                    # o simili. Se il tuo modello usa una sintassi diversa per generare
                    # l'immagine marchiata, modificala qui.
                    embed_result = pixelseal.embed(img_tensor)
                    
                    # Gestione dell'output in base a come Meta restituisce i dati
                    if isinstance(embed_result, dict) and "imgs" in embed_result:
                        wm_tensor = embed_result["imgs"]
                    else:
                        wm_tensor = embed_result
                
                # D. Salvataggio Immagine Watermarked
                # Rimuoviamo la dimensione del batch [1, ...] e torniamo a PIL
                wm_img_pil = TF.to_pil_image(wm_tensor.squeeze(0).cpu())
                wm_save_path = os.path.join(wm_out_dir, crop_filename)
                wm_img_pil.save(wm_save_path)
                
    print(f"\n🎉 Generazione completata! Dataset salvato nella cartella: {OUT_ROOT}")
    print("Struttura creata: dataset_minSize/ [train, val, test] / [clean_img, wm_img]")

if __name__ == "__main__":
    main()
