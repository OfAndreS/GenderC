#!/bin/bash
#=======================================================================
# DIRETIVAS DO SLURM
#=======================================================================

#SBATCH --job-name=GenderC_ExpFull        # Nome atualizado para facilitar
#SBATCH --partition=amd-512               # Partição correta
#SBATCH --ntasks=1                        # Número de tarefas
#SBATCH --cpus-per-task=12                # 12 CPUs (Seu Sweet Spot)
#SBATCH --mem=32G                         # 32GB RAM (Margem de segurança)
#SBATCH --time=1-00:00:00                 # TEMPO: 1 dia (Garante que não será cortado)

# --- Salvando Output ---
#SBATCH --output=resultados/resultado_%j.out
#SBATCH --error=erros/erro_%j.err
  
#=======================================================================
# COMANDOS DE EXECUÇÃO
#=======================================================================

echo "==============================================================="
echo "Job iniciado em: $(date)"
echo "Nó: $(hostname)"
echo "==============================================================="

# Navegue para o diretório do seu projeto
cd ~/SyncDesk/GenderC

# Inicialize o Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Ative o ambiente
conda activate DeepLearningAdvanced

# Execute o script Python
echo "Iniciando Experimento Científico Completo (Core)..."

# O Core orquestra tudo: gera dados (se faltar), treina e salva logs
python -m src.Core.core

echo "Execucao do .py finalizada."

echo "==============================================================="
echo "Job finalizado em: $(date)"
echo "==============================================================="