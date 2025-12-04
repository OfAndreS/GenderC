import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import ast
import os
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
# 2. FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
def parse_history(history_str):
    try:
        if pd.isna(history_str): return None
        history_list = ast.literal_eval(history_str)
        if isinstance(history_list, list) and len(history_list) > 0:
            return float(history_list[-1])
        return None
    except:
        return None

def plot_tradeoff_global_mean_legend(file_path):
    # Verificação
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo {file_path} não encontrado.")
        return

    df = pd.read_csv(file_path)

    # 3. PROCESSAMENTO
    df['final_train_acc'] = df['history_accuracy'].apply(parse_history)
    col_validacao = 'test_accuracy'
    df = df.dropna(subset=['final_train_acc', col_validacao])
    
    df['Overfitting_Gap'] = df['final_train_acc'] - df[col_validacao]

    # Cálculo da Média Global
    global_mean_gap = df['Overfitting_Gap'].mean()
    global_mean_acc = df[col_validacao].mean()

    # Ordenação Numérica das Labels
    col_hue = 'n_mfcc'
    if col_hue in df.columns:
        unique_values_int = sorted(df[col_hue].dropna().unique().astype(int))
        unique_categories = [str(x) for x in unique_values_int]
        df[col_hue] = df[col_hue].astype(str)
    else:
        unique_categories = ['Default']
        df['Default'] = 'Default'
        col_hue = 'Default'

    # 4. CONFIGURAÇÃO VISUAL
    palette_pastel = sns.color_palette("pastel", n_colors=len(unique_categories))
    map_pastel = dict(zip(unique_categories, palette_pastel))
    
    available_markers = ['o', 's', 'D', '^', 'v', 'X', 'P', '*'] 
    if len(unique_categories) > len(available_markers):
        available_markers = available_markers * (len(unique_categories) // len(available_markers) + 1)
    marker_map = dict(zip(unique_categories, available_markers[:len(unique_categories)]))

    summary_df = df.groupby(col_hue)[['Overfitting_Gap', col_validacao]].agg(['mean', 'std'])

    # 5. PLOTAGEM
    plt.figure(figsize=(12, 8))

    # A. CRUZ DA MÉDIA GLOBAL (Agora Transparente e com Label para Legenda)
    # Linha Vertical (Adicionamos label aqui para aparecer na legenda)
    plt.axvline(x=global_mean_gap, color='#555555', linestyle='--', 
                linewidth=1.5, alpha=0.4, zorder=1, label='Média Global')
    
    # Linha Horizontal (Sem label para não duplicar na legenda)
    plt.axhline(y=global_mean_acc, color='#555555', linestyle='--', 
                linewidth=1.5, alpha=0.4, zorder=1)
    
    # (Removido plt.text da média global conforme pedido)

    # B. Seta de Contexto
    plt.annotate('Direção Ideal\n(Alta Acurácia, Baixo Gap)', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 xytext=(0.25, 0.85), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5, alpha=0.3),
                 fontsize=10, color='gray', alpha=0.6, ha='center')

    # C. Runs Individuais (Fundo)
    # Importante: Definimos legend='auto' para o Seaborn gerar os handles das categorias
    sns.scatterplot(
        data=df,
        x='Overfitting_Gap',
        y=col_validacao,
        hue=col_hue,
        style=col_hue,
        hue_order=unique_categories,
        style_order=unique_categories,
        palette=map_pastel,
        markers=marker_map,
        s=90,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.5,
        zorder=3
    )

    # D. Loop de Médias (Desenha os pontos centrais)
    for category in unique_categories:
        if category not in summary_df.index: continue

        mean_x = summary_df.loc[category, ('Overfitting_Gap', 'mean')]
        std_x  = summary_df.loc[category, ('Overfitting_Gap', 'std')]
        mean_y = summary_df.loc[category, (col_validacao, 'mean')]
        std_y  = summary_df.loc[category, (col_validacao, 'std')]
        
        c_color = map_pastel[category] 
        c_marker = marker_map[category]
        
        # Barras de Erro
        plt.errorbar(
            x=mean_x, y=mean_y, xerr=std_x, yerr=std_y,
            fmt='none', ecolor='#444444', elinewidth=1.2, capsize=4, alpha=0.6, zorder=9
        )

        # Marcador Central
        plt.scatter(
            x=mean_x, y=mean_y, s=280, marker=c_marker,
            color=c_color, edgecolors='black', linewidth=2, zorder=10
        )
        
        # Label no Gráfico
        label_y = mean_y - std_y - 0.003 
        plt.text(
            x=mean_x, y=label_y, 
            s=f"MFCC-{category}", fontsize=9, fontweight='bold',
            color='#333333', ha='center', va='top', zorder=11,
            path_effects=[path_effects.withStroke(linewidth=3, foreground='white')]
        )

    # 6. FINALIZAÇÃO E LEGENDA
    plt.title('Trade-Off: Acurácia vs. Gap', fontsize=16, pad=20, weight='bold')
    plt.xlabel(r'Gap de Overfitting (Treino - Teste) $\leftarrow$ Melhor', fontsize=12)
    plt.ylabel(r'Acurácia de Validação $\uparrow$ Melhor', fontsize=12) 
    
    sns.despine(offset=10, trim=False)
    
    # AJUSTE DA LEGENDA
    # O Seaborn gera handles para o Hue. O axvline tem seu próprio handle.
    # Vamos pedir ao Matplotlib para juntar tudo.
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Para garantir que a ordem fique bonita (Categorias primeiro, Média Global por último ou vice-versa)
    # O axvline geralmente é adicionado primeiro se foi plotado primeiro.
    # Mas o Seaborn pode ter adicionado seus handles depois.
    
    # Vamos reordenar para colocar a "Média Global" no final da lista, se preferir
    # (Opcional: se quiser ordem específica, basta filtrar as listas handles/labels)
    
    plt.legend(handles=handles, labels=labels, title='Flexibilização', 
               bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()

    output_path = 'graphics/gComplex/TradeOff_Publication_Legend.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico final salvo em: {output_path}")

# Execução
file_path = 'output/resultados_experimento.csv'
plot_tradeoff_global_mean_legend(file_path)