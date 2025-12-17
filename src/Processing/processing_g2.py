import matplotlib
# Força backend não-interativo para evitar erros de display
matplotlib.use('Agg') 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.lines import Line2D

# 1. Carregar os dados
file_path = 'output/resultados_experimento_30_runs.pkl'

try:
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    # Dados dummy para teste caso o arquivo não exista no ambiente local
    print("AVISO: Usando dados dummy para teste.")
    data = {
        'n_mfcc': np.random.choice([5, 13, 20, 30, 40, 50], 180),
        'test_loss': np.random.uniform(0.8, 1.5, 180) # Valores típicos de loss
    }
    df = pd.DataFrame(data)

# Configurar Variável de Interesse
y_var = "test_loss"
y_label = "Loss (Perda)"
title = "Perda Final de Teste (Menor é Melhor)"

# 2. Configuração do Estilo e Paleta
sns.set_theme(style="whitegrid")

unique_mfccs = sorted(df['n_mfcc'].unique())
base_palette = ["#98D8C8", "#F7B7A3", "#B5C7D9", "#EBAFD2", "#C6E59D", "#F5E68C"]

# Ajuste dinâmico de cores
if len(unique_mfccs) > len(base_palette):
    extra_colors = ["#D3D3D3"] * (len(unique_mfccs) - len(base_palette))
    custom_palette = base_palette + extra_colors
else:
    custom_palette = base_palette[:len(unique_mfccs)]

# 3. Preparação da Figura
plt.figure(figsize=(12, 7))

# 4. Plotagem das Camadas

# Camada A: Gráfico de Violino
ax = sns.violinplot(
    x="n_mfcc", 
    y=y_var, 
    data=df, 
    palette=custom_palette, 
    hue="n_mfcc",
    legend=False,
    saturation=0.9, 
    linewidth=0, 
    inner=None, 
    density_norm="width"
)

# Camada B: Box Plot
sns.boxplot(
    x="n_mfcc", 
    y=y_var, 
    data=df, 
    width=0.15,
    boxprops={'zorder': 2, 'facecolor':'none', 'edgecolor':'#404040', 'linewidth': 1.5}, 
    whiskerprops={'color':'#404040', 'linewidth': 1.5},
    capprops={'color':'#404040', 'linewidth': 1.5},
    medianprops={'color':'#202020', 'linewidth': 2},
    showfliers=False,
    ax=ax
)

# Camada C: Swarm Plot
sns.swarmplot(
    x="n_mfcc", 
    y=y_var, 
    data=df, 
    color="white", 
    edgecolor="gray", 
    linewidth=0.8, 
    size=5, 
    alpha=0.9,
    ax=ax
)

# 5. Linha de Baseline (N=13)
if 13 in unique_mfccs:
    baseline_val = df[df['n_mfcc'] == 13][y_var].mean()
    label_text = f'Baseline (N=13): {baseline_val:.4f}'
else:
    baseline_val = df[y_var].mean()
    label_text = f'Média Global: {baseline_val:.4f}'

plt.axhline(y=baseline_val, color='#D9534F', linestyle='--', linewidth=2, alpha=0.8, zorder=0)

# 6. Legenda e Ajustes Finais
legend_element = [Line2D([0], [0], color='#D9534F', lw=2, linestyle='--', label=label_text)]
plt.legend(handles=legend_element, loc='upper right', fontsize=12, framealpha=0.9) # Mudei para upper right pois loss costuma ser menor, liberando espaço no topo

plt.title(title, fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Resolução Espectral (N_MFCC)", fontsize=13, labelpad=10)
plt.ylabel(y_label, fontsize=13, labelpad=10)

# Ajuste dinâmico dos limites Y (Loss varia muito, melhor não fixar hardcoded)
y_min = df[y_var].min()
y_max = df[y_var].max()
padding = (y_max - y_min) * 0.15
plt.ylim(bottom=max(0, y_min - padding), top=y_max + padding)

plt.tick_params(axis='both', which='major', labelsize=12)
sns.despine(left=True, bottom=False)
plt.grid(axis='y', linestyle=':', alpha=0.6)

# 7. Salvar
output_filename = "grafico_loss_recriado.png"
plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"GRÁFICO DE LOSS SALVO: {output_filename}")