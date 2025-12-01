#!/bin/bash
#=======================================================================
# DIRETIVAS DO SLURM
#=======================================================================
#SBATCH --job-name=Emotions_V100
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=resultados/treino_%j.out
#SBATCH --error=erros/treino_%j.err

echo "==============================================================="
echo "Job iniciado em: $(date)"
echo "Nó: $(hostname)"
echo "GPU ID: $CUDA_VISIBLE_DEVICES"
echo "==============================================================="

cd ~/SyncDesk/GenderC

# 1. BLINDAGEM DE AMBIENTE (NOVO & CRÍTICO)
# Isso impede que pacotes da sua pasta pessoal (.local) quebrem o Conda
export PYTHONNOUSERSITE=1

# 2. CARREGAR MÓDULOS (Ignora erros)
module purge
module load compilers/nvidia/cuda/11.7 || echo ">> Aviso: Modulo CUDA do sistema nao carregado."
module load gcc || echo ">> Aviso: Modulo GCC nao carregado."

# 3. ATIVAR AMBIENTE CONDA
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DeepLearningFinal

# 4. AUTO-REPARO DE BIBLIOTECAS
echo "--- Configurando Ambiente XLA/CUDA ---"

# A) Linkar libdevice
LIBDEVICE_FOUND=$(find $CONDA_PREFIX -name libdevice.10.bc | head -n 1)
if [ ! -z "$LIBDEVICE_FOUND" ]; then
    mkdir -p $CONDA_PREFIX/nvvm/libdevice
    ln -sf $LIBDEVICE_FOUND $CONDA_PREFIX/nvvm/libdevice/libdevice.10.bc
fi

# B) Linkar GCC
GCC_FOUND=$(find $CONDA_PREFIX/bin -name "*-gcc" | head -n 1)
if [ ! -z "$GCC_FOUND" ]; then
    ln -sf $GCC_FOUND $CONDA_PREFIX/bin/gcc
    ln -sf $GCC_FOUND $CONDA_PREFIX/bin/g++
fi

# 5. VARIÁVEIS DE AMBIENTE
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# 6. DIAGNÓSTICO E EXECUÇÃO
echo "--- Verificando ---"
# Verifica se o NumPy carregado é o correto (do Conda, não do .local)
python -c "import numpy; print(f'Numpy Path: {numpy.__file__}'); import tensorflow as tf; print(f'GPU: {len(tf.config.list_physical_devices(\"GPU\"))}')"

echo "--- Iniciando Treino ---"
python -m src.core.predictor
echo "--- Finalizado ---"