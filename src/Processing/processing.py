import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import ast
import os
import re
import numpy as np

# -----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DE ESTILO
# -----------------------------------------------------------------------------
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# -----------------------------------------------------------------------------
# 2. FUNÇÕES AUXILIARES BLINDADAS
# -----------------------------------------------------------------------------
def parse_history(history_data):
    """
    Extrai a última acurácia com robustez para diversos formatos.
    """
    try:
        # 1. TIPO ITERÁVEL (Lista, Numpy Array, Series)
        if isinstance(history_data, (list, np.ndarray, pd.Series)):
            if len(history_data) == 0: return None
            flat = np.ravel(history_data)
            if len(flat) > 0: return float(flat[-1])
            return None

        # 2. VALORES NULOS
        if pd.isna(history_data): return None
        
        # 3. STRINGS (Parsing de formatos CSV ou Numpy-string)
        if isinstance(history_data, str):
            s = history_data.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                        return float(parsed[-1])
                except:
                    pass
                
                try:
                    content = s[1:-1].strip()
                    parts = [x for x in re.split(r'\s+', content) if x]
                    if len(parts) > 0:
                        return float(parts[-1])
                except:
                    pass

        # 4. VALOR NUMÉRICO DIRETO
        return float(history_data)
            
    except Exception:
        return None

def plot_tradeoff_global_mean_legend(file_path):
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo {file_path} não encontrado.")
        return

    # -------------------------------------------------------------------------
    # 3. CARREGAMENTO
    # -------------------------------------------------------------------------
    print(f"--> Carregando: {file_path}")
    try:
        if file_path.endswith('.pkl'):
            df = pd.read_pickle(file_path)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"ERRO CRÍTICO ao ler arquivo: {e}")
        return

    col_validacao = 'test_accuracy'
    
    # Processa 'final_train_acc'
    if 'final_train_acc' not in df.columns:
        if 'history_accuracy' in df.columns:
            print("--> Calculando 'final_train_acc'...")
            df['final_train_acc'] = df['history_accuracy'].apply(parse_history)
        else:
            print("ERRO: Coluna 'history_accuracy' ausente.")
            return

    # Limpeza
    df_clean = df.dropna(subset=['final_train_acc', col_validacao]).copy()
    
    if len(df_clean) == 0:
        print("\n=== ERRO CRÍTICO: DATAFRAME VAZIO ===")
        return

    df = df_clean 

    # Métricas
    df['Overfitting_Gap'] = df['final_train_acc'] - df[col_validacao]
    global_mean_gap = df['Overfitting_Gap'].mean()
    global_mean_acc = df[col_validacao].mean()

    # -------------------------------------------------------------------------
    # 4. CONFIGURAÇÃO DE CATEGORIAS
    # -------------------------------------------------------------------------
    col_hue = 'n_mfcc'
    if col_hue in df.columns:
        try:
            # Ordenação numérica
            vals = sorted(df[col_hue].dropna().unique().astype(int))
            unique_categories = [str(x) for x in vals]
            df[col_hue] = df[col_hue].astype(int).astype(str)
        except:
            unique_categories = sorted(df[col_hue].dropna().astype(str).unique())
            df[col_hue] = df[col_hue].astype(str)
    else:
        col_hue = 'Geral'
        df[col_hue] = 'All'
        unique_categories = ['All']

    # -------------------------------------------------------------------------
    # 5. VISUALIZAÇÃO: AZUL -> VERMELHO (Coolwarm)
    # -------------------------------------------------------------------------
    # A paleta "coolwarm" vai do Azul (Cool) para o Vermelho (Warm).
    # Como as categorias estão ordenadas numericamente (5, 7, 13... 50),
    # 5 será Azul e 50 será Vermelho.
    
    n_cats = len(unique_categories)
    palette_custom = sns.color_palette("coolwarm", n_colors=n_cats)

    map_colors = dict(zip(unique_categories, palette_custom))
    
    markers = ['o', 's', 'D', '^', 'v', 'X', 'P', '*'] 
    if len(unique_categories) > len(markers):
        markers = markers * (len(unique_categories)//len(markers) + 1)
    map_markers = dict(zip(unique_categories, markers[:len(unique_categories)]))

    summary_df = df.groupby(col_hue)[['Overfitting_Gap', col_validacao]].agg(['mean', 'std'])

    # -------------------------------------------------------------------------
    # 6. PLOTAGEM
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))

    # Linhas de Referência
    plt.axvline(x=global_mean_gap, color='#555555', linestyle='--', linewidth=1.5, alpha=0.4, label='Média Global')
    plt.axhline(y=global_mean_acc, color='#555555', linestyle='--', linewidth=1.5, alpha=0.4)

    plt.annotate('Direção Ideal\n(Alta Acurácia, Baixo Gap)', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 xytext=(0.25, 0.85), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, alpha=0.3),
                 fontsize=10, color='gray', alpha=0.6, ha='center')

    # Scatter de Fundo (Runs)
    try:
        sns.scatterplot(
            data=df,
            x='Overfitting_Gap',
            y=col_validacao,
            hue=col_hue,
            style=col_hue,
            hue_order=unique_categories,
            style_order=unique_categories,
            palette=map_colors,
            markers=map_markers,
            s=90,
            edgecolor='white',
            linewidth=0.5,
            alpha=0.6, # Transparência média
            zorder=3
        )
    except Exception as e:
        print(f"Aviso no Scatter: {e}")

    # Médias e Erros (Pontos Centrais)
    for cat in unique_categories:
        if cat not in summary_df.index: continue

        mx = summary_df.loc[cat, ('Overfitting_Gap', 'mean')]
        sx = summary_df.loc[cat, ('Overfitting_Gap', 'std')]
        my = summary_df.loc[cat, (col_validacao, 'mean')]
        sy = summary_df.loc[cat, (col_validacao, 'std')]
        
        sx = 0 if pd.isna(sx) else sx
        sy = 0 if pd.isna(sy) else sy

        c_col = map_colors[cat]
        c_mark = map_markers[cat]

        # Barra de Erro
        plt.errorbar(x=mx, y=my, xerr=sx, yerr=sy, fmt='none', ecolor='#444444', elinewidth=1.2, capsize=4, alpha=0.6, zorder=9)
        
        # Ponto Central
        plt.scatter(x=mx, y=my, s=280, marker=c_mark, color=c_col, edgecolors='black', linewidth=2, zorder=10)
        
        # Rótulo
        lbl = f"MFCC-{cat}" if col_hue == 'n_mfcc' else cat
        plt.text(x=mx, y=my - sy - 0.003, s=lbl, fontsize=9, fontweight='bold', color='#333333', ha='center', va='top', zorder=11, path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])

    # Finalização
    plt.title('Trade-Off: Acurácia vs. Gap (Azul/Baixo -> Vermelho/Alto)', fontsize=16, pad=20, weight='bold')
    plt.xlabel(r'Gap de Overfitting (Treino - Teste) $\leftarrow$ Melhor', fontsize=12)
    plt.ylabel(r'Acurácia de Validação $\uparrow$ Melhor', fontsize=12) 
    sns.despine(offset=10, trim=False)
    
    # Legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    sorted_labs = []
    if 'Média Global' in by_label: sorted_labs.append('Média Global')
    sorted_labs.extend([c for c in unique_categories if c in by_label])
    sorted_hands = [by_label[l] for l in sorted_labs]

    plt.legend(sorted_hands, sorted_labs, title='Nº MFCCs', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.tight_layout()

    out_path = 'graphics/gComplex/TradeOff_BlueToRed_30runs.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"--> Sucesso! Gráfico salvo em: {out_path}")

if __name__ == "__main__":
    file_path = 'output/resultados_experimento_30_runs.pkl'
    plot_tradeoff_global_mean_legend(file_path)