#!/bin/bash
# Script para criar a estrutura atualizada do projeto
# Execute este script DENTRO do diretório raiz do seu projeto (ex: ./meu_novo_projeto/)

echo "Criando diretórios..."

# Criar as pastas principais
mkdir -p configs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p notebooks

# Criar as subpastas de src (com a primeira letra maiúscula conforme o diagrama)
mkdir -p src/Core
mkdir -p src/Training
mkdir -p src/Processing
mkdir -p src/Shared

echo "Criando arquivos..."

# Arquivos da raiz e configurações
touch requirements.txt
touch configs/params.yml
touch notebooks/exploracao_de_dados.ipynb

# Arquivos do módulo Core
touch src/Core/__init__.py
touch src/Core/core.py

# Arquivos do módulo Training
touch src/Training/__init__.py
touch src/Training/training.py

# Arquivos do módulo Processing
touch src/Processing/__init__.py
touch src/Processing/processing.py

# Arquivos do módulo Shared
touch src/Shared/__init__.py
touch src/Shared/utils.py

# Listar a estrutura criada para verificação
echo "Estrutura de pastas e arquivos criada com sucesso:"
ls -R
