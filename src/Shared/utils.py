import os
import json
import numpy as np

def printHead():
    print("\n\n|  * * * * * * * * * * * * * * * * * *\n\n")

def getData():

    json_path = "data/processed"
    json_file = "data_10_segments.json"
    
    DataInDict = LoadDataFromJson(json_path, json_file)

    if not DataInDict:
        print("| ERRO CRÍTICO: Não foi possível carregar o dicionário de dados.")
        return [], [], [], []

    getMapping = DataInDict.get("mapping", [])
    getLabels = DataInDict.get("labels", [])
    getMFCC = DataInDict.get("mfcc", [])
    getFiles = DataInDict.get("files", []) 

    return np.array(getMFCC), np.array(getLabels), np.array(getMapping), np.array(getFiles)


def SaveDataInJson(data_dict, dirPath, jsonFileName):
    printHead()
    fullJsonPath = os.path.join(dirPath, jsonFileName)

    try:
        with open(fullJsonPath, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
            
        print(f"| Dados salvos - {fullJsonPath}")
        printHead()

    except IOError as e:
        print(f"| ERRO: IOError - {fullJsonPath} - {e}")
        printHead()
    except TypeError as e:
        print(f"| ERRO: Os dados fornecidos não são serializáveis para JSON - {e}")
        printHead()


def LoadDataFromJson(dirPath, jsonFileName):
    printHead()
    fullJsonPath = os.path.join(dirPath, jsonFileName)
    loadedData = {} 

    try:
        with open(fullJsonPath, 'r', encoding='utf-8') as f:
            loadedData = json.load(f)
            
        print(f"| Dados carregados - {fullJsonPath}")
        printHead()

    except FileNotFoundError:
        print(f"| ERRO: Arquivo não encontrado - {fullJsonPath}")
        printHead()
    except json.JSONDecodeError as e:
        print(f"| ERRO: O arquivo não é um JSON válido - {fullJsonPath} - {e}")
        printHead()
    except IOError as e:
        print(f"| ERRO: IOError ao ler o arquivo - {fullJsonPath} - {e}")
        printHead()

    return loadedData