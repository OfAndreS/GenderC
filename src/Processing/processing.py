import os
import librosa
import math
import numpy as np
from src.Shared import utils

N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 22050
N_MFCC = 20

DURATION = 30 
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
EXPECTED_NUM_MFCC_VECTORS = math.ceil(SAMPLES_PER_TRACK / HOP_LENGTH) 

def ConvertWavToMfcc(fullAudioFilePath : str):
    try:
        signal, sr = librosa.load(fullAudioFilePath, sr=SAMPLE_RATE)
        
        if len(signal) < SAMPLES_PER_TRACK:
            pad_width = SAMPLES_PER_TRACK - len(signal)
            signal = np.pad(signal, (0, pad_width), mode='constant')
        else:
            signal = signal[:SAMPLES_PER_TRACK]
        
        MFCCs = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
        
        if MFCCs.shape[1] > EXPECTED_NUM_MFCC_VECTORS:
            MFCCs = MFCCs[:, :EXPECTED_NUM_MFCC_VECTORS]
        elif MFCCs.shape[1] < EXPECTED_NUM_MFCC_VECTORS:
             pad_width = EXPECTED_NUM_MFCC_VECTORS - MFCCs.shape[1]
             MFCCs = np.pad(MFCCs, ((0, 0), (0, pad_width)), mode='constant')

        mean = np.mean(MFCCs, axis=1, keepdims=True)
        std = np.std(MFCCs, axis=1, keepdims=True)
        normalized_mfcc = (MFCCs - mean) / (std + 1e-6)

        return normalized_mfcc
        
    except Exception as e:
        print(f"| ERRO ao processar {fullAudioFilePath}: {e}")
        return None


def LoadData(rootDirPath):
    listOfData = []
    listOfLabel = []

    for root, dirs, files in os.walk(rootDirPath):
        
        current_label = os.path.basename(root)
        
        if root == rootDirPath:
            continue

        for index, fileName in enumerate(files):
            
            if fileName.endswith(".wav"):
                print(f"| [ {index} ] - Processando {fileName}...")
                fullAudioFilePath = os.path.join(root, fileName)
                
                audioConverted = ConvertWavToMfcc(fullAudioFilePath)
                
                if audioConverted is not None:

                    listOfData.append(audioConverted.tolist())
                    listOfLabel.append(current_label)

    return listOfData, listOfLabel


def ProcessingData(listOfData, listOfLabel):
    if len(listOfData) != len(listOfLabel):
        return {}
    
    unique_labels = sorted(list(set(listOfLabel)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    integer_labels = [label_to_index[label] for label in listOfLabel]

    dataset_dictionary = {
        "mapping": unique_labels,
        "labels": integer_labels,
        "MFCC": listOfData
    }

    return dataset_dictionary

if __name__ == "__main__":
    raw_path = "data/raw/genres_simple"
    
    if os.path.exists(raw_path):
        print("| Iniciando processamento com tamanho FIXO...")
        data, labels = LoadData(raw_path)
        processedData = ProcessingData(data, labels)
        utils.SaveDataInJson(processedData, "data/processed", "go_Simple.json")
        print("| Concluído!")