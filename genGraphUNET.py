import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches  # <--- Importante per la legenda custom

# Usa il file generato dallo script di test della U-Net
NOME_FILE_CSV = "risultati_watermark_UNET.csv"
print(f"Lettura dati da {NOME_FILE_CSV}...")
df = pd.read_csv(NOME_FILE_CSV, sep=';')

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7)) 

# Colori: Verde per Pulite, Azzurro per Watermarked, Rosso per Post U-Net
colori = {
    "Pulita": "limegreen",
    "Watermarked": "dodgerblue",
    "Attaccata": "crimson"
}

# Creazione dell'istogramma impilato
sns.histplot(
    data=df, 
    x='bit accuracy', 
    hue='stato',
    palette=colori,
    hue_order=['Pulita', 'Watermarked', 'Attaccata'], 
    multiple='stack',
    binwidth=0.01, 
    alpha=0.9, 
    edgecolor='black',
    legend=False
)

# Aggiorniamo il nome dell'attacco per il titolo e il file
NOME_ATTACCO = "U-Net Denoising"

plt.title(f'Istogramma Bit Accuracy: Immagini Pulite, Watermarked e Post U-Net', fontsize=15, pad=15)
plt.xlabel('Bit Accuracy (0.0 = 0%, 1.0 = 100%)', fontsize=12)
plt.ylabel('Numero di Immagini', fontsize=12)

# ==========================================
# CREAZIONE MANUALE DELLA LEGENDA (Aggiornata per la U-Net)
# ==========================================
legend_elements = [
    mpatches.Patch(facecolor='limegreen', edgecolor='black', alpha=0.9, label='Pulita (Originale, Riferimento)'),
    mpatches.Patch(facecolor='dodgerblue', edgecolor='black', alpha=0.9, label='Watermarked (Segnale Intatto al 100%)'),
    mpatches.Patch(facecolor='crimson', edgecolor='black', alpha=0.9, label='Post U-Net (Ripulita dall\'Autoencoder)')
]

# Inseriamo la legenda dentro il grafico in alto a sinistra
plt.legend(handles=legend_elements, loc='upper left', title="Stato Immagine", fontsize=11, title_fontsize=12, frameon=True, shadow=True)

plt.tight_layout()

# Salvataggio dell'immagine
NOME_IMMAGINE_SALVATA = f'{NOME_ATTACCO.replace(" ", "_")}_istogramma.png'
plt.savefig(NOME_IMMAGINE_SALVATA, dpi=300)
print(f"✅ Grafico generato e salvato come: {NOME_IMMAGINE_SALVATA}")

plt.show()