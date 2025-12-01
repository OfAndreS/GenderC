import os
import librosa
import math
import json

SAMPLE_RATE = 22050
TRACK_DURATION = 30 
SAMPLES_POR_TRACK = SAMPLE_RATE * TRACK_DURATION

DATASET_PATH = "data/raw/genres_original"
JSON_DIR = "data/processed" 

LIST_OF_N_MFCC = [13, 20, 30, 40, 50]
LIST_OF_MFCC_NAMES = ["data_mfcc_13.json", "data_mfcc_20.json", "data_mfcc_30.json", "data_mfcc_40.json", "data_mfcc_50.json"]

def SaveMfcc(datasetPath, jsonPath, nMfcc, nFft=2048, hopLength=512, numSegments=10):

    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        "files": []
    }

    samplesPerSegment = int(SAMPLES_POR_TRACK / numSegments)
    numMfccVectorsPerSegment = math.ceil(samplesPerSegment / hopLength)

    print(f"| Salvando MFCCs (n_mfcc={nMfcc}) em: {jsonPath}")

    for i, (dirPath, dirNames, fileNames) in enumerate(os.walk(datasetPath)):
        
        if dirPath is not datasetPath:
            semanticLabel = os.path.basename(dirPath)
            data["mapping"].append(semanticLabel)
            print(f"|   Processando Gênero: {semanticLabel}")

            for f in fileNames:
                filePath = os.path.join(dirPath, f)
                try:
                    signal, sr = librosa.load(filePath, sr=SAMPLE_RATE)

                    for d in range(numSegments):
                        start = samplesPerSegment * d
                        finish = start + samplesPerSegment

                        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sr, n_mfcc=nMfcc, n_fft=nFft, hop_length=hopLength)
                        mfcc = mfcc.T

                        if len(mfcc) == numMfccVectorsPerSegment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            data["files"].append(f)
                            
                except Exception as e:
                    print(f"| ERRO ao ler {filePath}: {e}")

    os.makedirs(os.path.dirname(jsonPath), exist_ok=True)
    
    with open(jsonPath, "w", encoding='utf-8') as fp:
        json.dump(data, fp, indent=4)
        
    print(f"| Sucesso! Arquivo gerado: {jsonPath}\n")


def GenerateMfccDatasets(datasetPath, outputDir, listOfnMfcc, listOfMfccNames):

    if len(listOfnMfcc) != len(listOfMfccNames):
        print("| ERRO: Quantidade de itens incompatível entre as listas!")
        return

    for i, nMfcc in enumerate(listOfnMfcc):
        fileName = listOfMfccNames[i]
        
        fullJsonPath = os.path.join(outputDir, fileName)
        
        SaveMfcc(datasetPath, fullJsonPath, nMfcc)


if __name__ == "__main__":
    GenerateMfccDatasets(DATASET_PATH, JSON_DIR, LIST_OF_N_MFCC, LIST_OF_MFCC_NAMES)