import os
import gc
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from src.Shared import utils
from src.Training import training
from src.Processing import processing

listOfNMfcc = [5, 7, 13, 20, 30, 40, 50]
numRuns = 30
# MUDANÇA 1: Extensão .pkl
resultFile = "output/resultados_experimento_30_runs.pkl"
seedsFile = "configs/seeds_config.json"

rawDataPath = "data/raw/genres_original"
processedDir = "data/processed"

def GenerateFixedSeeds(n, filePath):
    if os.path.exists(filePath):
        print(f"|  CARREGANDO SEEDS EXISTENTES DE: {filePath} ")
        try:
            with open(filePath, 'r') as f:
                seeds = json.load(f)
            if len(seeds) != n:
                print(f"| AVISO: O arquivo tem {len(seeds)} seeds, mas o experimento pede {n}.")
            return seeds
        except Exception as e:
            print(f"| ERRO ao ler seeds: {e}")

    print(f"|  GERANDO NOVAS SEEDS E SALVANDO EM: {filePath} ")
    rng = np.random.default_rng(42)
    seeds = rng.integers(low=1, high=10000, size=n).tolist()
    
    try:
        with open(filePath, 'w') as f:
            json.dump(seeds, f)
    except Exception as e:
        print(f"| ERRO ao salvar seeds: {e}")
    return seeds

def LoadSpecificJson(jsonPath):
    try:
        data = utils.LoadDataFromJson(os.path.dirname(jsonPath), os.path.basename(jsonPath))
        x = np.array(data.get("mfcc", []))
        y = np.array(data.get("labels", []))
        groups = np.array(data.get("files", []))
        return x, y, groups
    except Exception as e:
        print(f"| ERRO ao carregar {jsonPath}: {e}")
        return None, None, None

def RunExperiment():
    os.makedirs(os.path.dirname(resultFile), exist_ok=True)
    os.makedirs(os.path.dirname(seedsFile), exist_ok=True)

    seedsList = GenerateFixedSeeds(numRuns, seedsFile)
    allResults = []

    for nMfcc in listOfNMfcc:
        print(f"\n| INICIANDO RODADA PARA N_MFCC = {nMfcc}")
        
        jsonName = f"data_mfcc_{nMfcc}.json"
        fullJsonPath = os.path.join(processedDir, jsonName)
        
        if not os.path.exists(fullJsonPath):
            processing.SaveMfcc(rawDataPath, fullJsonPath, nMfcc=nMfcc, numSegments=10)
        
        x, y, groups = LoadSpecificJson(fullJsonPath)
        
        if x is None or len(x) == 0:
            continue

        for i, seed in enumerate(seedsList):
            print(f"| > Execução {i+1}/{numRuns} (Seed: {seed})")
            
            tf.keras.backend.clear_session()
            gc.collect()
            tf.random.set_seed(seed)
            np.random.seed(seed)
            
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
                trainIdx, testIdx = next(gss.split(x, y, groups))
                
                xTrain, xTest = x[trainIdx], x[testIdx]
                yTrain, yTest = y[trainIdx], y[testIdx]
                groupsTrain = groups[trainIdx]

                gssVal = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                tIdx, vIdx = next(gssVal.split(xTrain, yTrain, groupsTrain))

                xTrainFinal, xVal = xTrain[tIdx], xTrain[vIdx]
                yTrainFinal, yVal = yTrain[tIdx], yTrain[vIdx]

                xTrainFinal = xTrainFinal[..., np.newaxis]
                xVal = xVal[..., np.newaxis]
                xTest = xTest[..., np.newaxis]

            except Exception as e:
                print(f"| ERRO no Split: {e}")
                continue

            inputShape = (xTrainFinal.shape[1], xTrainFinal.shape[2], 1)
            model = training.BuildModelOptimized(inputShape)
            
            startTime = time.time()
            history = training.TrainModel(model, xTrainFinal, yTrainFinal, xVal, yVal)
            endTime = time.time()
            trainDuration = endTime - startTime
            
            loss, acc = model.evaluate(xTest, yTest, verbose=0)
            print(f"| Resultado: Acc = {acc:.4f} | Tempo: {trainDuration:.2f}s")

            # MUDANÇA 2: Com Pickle, podemos salvar objetos NumPy diretamente (embora converter seja boa prática)
            allResults.append({
                "n_mfcc": nMfcc,
                "run_id": i + 1,
                "seed": int(seed),
                "test_accuracy": float(acc),
                "test_loss": float(loss),
                "epochs_trained": int(len(history.history['loss'])),
                "train_duration": float(trainDuration),
                "history_loss": history.history['loss'],         # Salva direto!
                "history_accuracy": history.history['accuracy'], # Salva direto!
                "history_val_loss": history.history['val_loss'],
                "history_val_accuracy": history.history['val_accuracy']
            })

            # MUDANÇA 3: Salvar como Pickle (.pkl)
            try:
                pd.DataFrame(allResults).to_pickle(resultFile)
            except Exception as e:
                print(f"| ERRO AO SALVAR PKL: {e}")

    print(f"| Resultados salvos em: {resultFile}")

if __name__ == "__main__":
    RunExperiment()