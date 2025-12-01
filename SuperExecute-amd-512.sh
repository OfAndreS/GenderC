#!/bin/bash
#=======================================================================
# DIRETIVAS DO SLURM
#=======================================================================

#SBATCH --job-name=EmotionsPatterns         # Nome do job para fácil identificação
#SBATCH --partition=amd-512                 # Partição correta
#SBATCH --ntasks=1                          # Número de tarefas
#SBATCH --cpus-per-task=12                   # Aumentado para 4 CPUs
#SBATCH --mem=16G                           # Aumentado para 16GB de RAM (aumento significativo)
#SBATCH --time=0-01:00:00                   # Aumentado para 1 hora

# --- Salvando Output ---
#SBATCH --output=resultados/resultado_%j.out
#SBATCH --error=erros/erro_%j.err
  
#=======================================================================
# COMANDOS DE EXECUÇÃO
#=======================================================================

echo "==============================================================="
echo "Job iniciado com mais recursos em: $(date)"
echo "Executando no no: $(hostname)"
echo "Arquivos de saida serao salvos em 'resultados/'"
echo "Arquivos de erro serao salvos em 'erros/'"
echo "==============================================================="

# Navegue para o diretório do seu projeto
cd ~/SyncDesk/GenderC

# Inicialize o Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Ative o ambiente
conda activate DeepLearningAdvanced

# Execute o script Python
echo "Iniciando execucao do app.py..."
# python -m src.Training.training
python -m src.Processing.processing
# python -m src.Core.core
echo "Execucao do .py finalizada."

echo "==============================================================="
echo "Job finalizado em: $(date)"
echo "==============================================================="
