import os
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

resultFile = "resultados_experimento.csv"
outputDir = "graphics_pastel"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['axes.titlepad'] = 20 
plt.rcParams['axes.labelpad'] = 15
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.facecolor'] = 'white'

PASTEL_PALETTE = "Set2"

def LoadExperimentData(csvPath):
    if not os.path.exists(csvPath):
        print(f"| ERRO: Ficheiro não encontrado: {csvPath}")
        return None

    df = pd.read_csv(csvPath)

    listCols = ['history_loss', 'history_accuracy', 'history_val_loss', 'history_val_accuracy']
    for col in listCols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return df

def PlotDistribution(df, xCol, yCol, title, ylabel, outputDir, filename, baseline=None, invertBest=False):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    uniqueX = sorted(df[xCol].unique())
    
    sns.violinplot(
        data=df, 
        x=xCol, 
        y=yCol, 
        order=uniqueX,
        palette=PASTEL_PALETTE, 
        hue=xCol,
        legend=False,
        inner=None,
        alpha=0.6,
        linewidth=0,
        ax=ax
    )
    
    sns.boxplot(
        data=df, 
        x=xCol, 
        y=yCol, 
        order=uniqueX,
        width=0.15, 
        boxprops={'zorder': 2, 'facecolor':'none', 'linewidth': 1.5},
        whiskerprops={'color':'#333333', 'linewidth': 1.5},
        capprops={'color':'#333333', 'linewidth': 1.5},
        medianprops={'color':'#333333', 'linewidth': 2.5, 'solid_capstyle': 'butt'},
        showfliers=False,
        ax=ax
    )
    
    sns.swarmplot(
        data=df, 
        x=xCol, 
        y=yCol, 
        order=uniqueX,
        color="white", 
        edgecolor="#333333", 
        linewidth=0.6,
        size=6.5,
        alpha=0.9,
        ax=ax
    )
    
    if baseline is not None:
        color = '#e74c3c' if not invertBest else '#2ecc71'
        label = f'Baseline (N=13): {baseline:.4f}'
        ax.axhline(y=baseline, color=color, linestyle='--', alpha=0.8, linewidth=2, label=label, zorder=0)
        ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.95, fancybox=True)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Resolução Espectral (N_MFCC)', fontweight='medium')
    ax.set_ylabel(ylabel, fontweight='medium')
    ax.grid(True, axis='y', linestyle=':', color='gray', alpha=0.4)
    sns.despine(left=True)
    
    outputPath = os.path.join(outputDir, filename)
    plt.savefig(outputPath, bbox_inches='tight')
    plt.close()
    print(f"| Gráfico Pastel Salvo: {outputPath}")

def PlotMetricEvolutionComparison(df, metricKey, title, ylabel, outputDir, filename):
    plt.figure(figsize=(14, 8))
    
    uniqueMfccs = sorted(df['n_mfcc'].unique())
    colors = sns.color_palette(PASTEL_PALETTE, n_colors=len(uniqueMfccs))

    for i, nMfcc in enumerate(uniqueMfccs):
        subset = df[df['n_mfcc'] == nMfcc]
        metricValues = subset[metricKey].tolist()
        
        minLen = min(len(x) for x in metricValues)
        metricValues = [x[:minLen] for x in metricValues]
        
        metricValues = np.array(metricValues)
        
        meanVals = np.mean(metricValues, axis=0)
        stdVals = np.std(metricValues, axis=0)
        epochs = range(1, len(meanVals) + 1)

        plt.plot(epochs, meanVals, label=f'N={nMfcc}', color=colors[i], linewidth=3, alpha=0.9)
        plt.fill_between(epochs, meanVals - stdVals, meanVals + stdVals, color=colors[i], alpha=0.12)

    plt.title(title, fontweight='bold')
    plt.xlabel('Épocas', fontweight='medium')
    plt.ylabel(ylabel, fontweight='medium')
    plt.legend(title="N_MFCC", loc='best', frameon=True, facecolor='white', fancybox=True)
    plt.grid(True, linestyle=':', color='gray', alpha=0.4)
    sns.despine()
    
    outputPath = os.path.join(outputDir, filename)
    plt.savefig(outputPath, bbox_inches='tight')
    plt.close()
    print(f"| Gráfico Pastel Salvo: {outputPath}")

def PlotGeneralizationGap(df, outputDir, filename):
    plt.figure(figsize=(12, 7))
    
    df['finalTrainAcc'] = df['history_accuracy'].apply(lambda x: x[-1])
    df['generalizationGap'] = df['finalTrainAcc'] - df['test_accuracy']
    
    ax = sns.barplot(
        data=df, 
        x="n_mfcc", 
        y="generalizationGap", 
        hue="n_mfcc", 
        palette=PASTEL_PALETTE, 
        legend=False,
        capsize=0.15,
        err_kws={'linewidth': 2, 'color': '#555555'},
        alpha=0.85,
        edgecolor=".2"
    )
    
    plt.title("Análise de Overfitting (Generalization Gap)", fontweight='bold')
    plt.ylabel("Diferença (Treino - Teste)", fontweight='medium')
    plt.xlabel("N_MFCC", fontweight='medium')
    plt.grid(True, axis='y', linestyle=':', color='gray', alpha=0.4)
    sns.despine(left=True)
    
    plt.axhline(0, color='gray', linewidth=1)

    outputPath = os.path.join(outputDir, filename)
    plt.savefig(outputPath, bbox_inches='tight')
    plt.close()
    print(f"| Gráfico Pastel Salvo: {outputPath}")

def GenerateThesisGraphics(inputCsv, outputDir):

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    print(f"| Carregando dados de: {inputCsv}")
    df = LoadExperimentData(inputCsv)

    if df is None:
        return

    print("| Gerando gráficos estéticos para tese...")

    baselineAcc = df[df['n_mfcc'] == 13]['test_accuracy'].mean()
    PlotDistribution(
        df,
        xCol='n_mfcc',
        yCol='test_accuracy',
        title='Acurácia Final de Teste (Maior é Melhor)',
        ylabel='Acurácia',
        outputDir=outputDir,
        filename='1_Thesis_Accuracy_Dist_Pastel.png',
        baseline=baselineAcc
    )

    baselineLoss = df[df['n_mfcc'] == 13]['test_loss'].mean()
    PlotDistribution(
        df,
        xCol='n_mfcc',
        yCol='test_loss',
        title='Perda Final de Teste (Menor é Melhor)',
        ylabel='Loss (Entropia Cruzada)',
        outputDir=outputDir,
        filename='2_Thesis_Loss_Dist_Pastel.png',
        baseline=baselineLoss,
        invertBest=True
    )

    PlotDistribution(
        df,
        xCol='n_mfcc',
        yCol='epochs_trained',
        title='Esforço de Convergência',
        ylabel='Épocas Treinadas',
        outputDir=outputDir,
        filename='3_Thesis_Convergence_Epochs_Pastel.png'
    )

    PlotGeneralizationGap(
        df,
        outputDir=outputDir,
        filename='4_Thesis_Overfitting_Gap_Pastel.png'
    )

    PlotMetricEvolutionComparison(
        df, 
        metricKey='history_val_accuracy', 
        title='Dinâmica de Aprendizado Comparativa', 
        ylabel='Acurácia de Validação', 
        outputDir=outputDir, 
        filename='5_Thesis_Learning_Dynamics_Pastel.png'
    )

    print("| Processo estético concluído.")

if __name__ == "__main__":
    GenerateThesisGraphics(resultFile, outputDir)