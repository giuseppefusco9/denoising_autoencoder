import torch
import pandas as pd
import matplotlib.pyplot as plt
from conv_ae_model import ConvAutoencoderDenoise

# ==========================================
# 1. INIZIALIZZAZIONE DEL MODELLO
# ==========================================
print("Caricamento architettura in corso...")
model = ConvAutoencoderDenoise(in_channels=3, out_channels=3)

# ==========================================
# 2. ESTRAZIONE PESI E STRUTTURA
# ==========================================
data = []
total_params = 0

# Iteriamo attraverso tutti i parametri della rete
for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue
    
    # Puliamo il nome del layer per renderlo più leggibile
    layer_name = name.replace(".weight", " (Pesi)").replace(".bias", " (Bias)")
    
    # Forma del tensore (es. [64, 3, 3, 3] = filtri, canali, altezza, larghezza)
    shape = str(list(parameter.shape))
    
    # Calcolo esatto dei parametri matematici per questo strato
    params = parameter.numel()
    
    data.append([layer_name, shape, f"{params:,}"])
    total_params += params

# Aggiungiamo la riga finale con il conteggio totale
data.append(["TOTALE COMPLESSIVO", "-", f"{total_params:,}"])

# ==========================================
# 3. STAMPA A SCHERMO
# ==========================================
df = pd.DataFrame(data, columns=["Layer (Nome Parametro)", "Forma del Tensore", "Numero di Pesi"])
print("\n" + "="*70)
print(df.to_string(index=False))
print("="*70 + "\n")

# ==========================================
# 4. GENERAZIONE IMMAGINE (Per la Tesi)
# ==========================================
print("Generazione immagine della tabella in corso...")

# Calcoliamo un'altezza dinamica in base a quanti strati ha la rete
fig_height = len(df) * 0.35 + 1.5
fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)
ax.axis('tight')
ax.axis('off')

# Creazione della tabella grafica
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0) # Allarga le celle per renderle ben spaziate

# Formattazione e Colori (Stile Accademico)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        # Intestazione
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_facecolor('#2c3e50')
    elif row == len(df):
        # Riga del totale
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e8ecef')
    else:
        # Righe alternate per maggiore leggibilità
        if row % 2 == 0:
            cell.set_facecolor('#f8f9fa')

plt.title("Struttura Architetturale: Parametri del ConvAutoencoder", fontweight="bold", fontsize=14, pad=20)
plt.tight_layout()

NOME_FILE = "Struttura_Pesi_ConvAE.png"
plt.savefig(NOME_FILE, bbox_inches='tight')
plt.close()

print(f"✅ Finito! L'immagine della struttura è stata salvata come: {NOME_FILE}")