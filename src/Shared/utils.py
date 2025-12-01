import os
import json
import numpy as np

def printHead():
    print("\n\n|  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *\n\n")


def getData():
    DataInDict = LoadDataFromJson("data/processed", "go_simple.json")

    getMapping = DataInDict["mapping"]
    getLabels = DataInDict["labels"]
    getMFCC = DataInDict["MFCC"]

    return getMFCC, getLabels, getMapping


def SaveDataInJson(listOfDataDict : list[dict], dirPath : str, jsonFileName : str):
    
    printHead()

    fullJsonPath = os.path.join(dirPath, jsonFileName)

    try:
        with open(fullJsonPath, 'w', encoding='utf-8') as f:

            json.dump(listOfDataDict, f, indent=4, ensure_ascii=False)
            
        print(f"| Dados salvos - {fullJsonPath}")
        printHead()

    except IOError as e:

        print(f"| ERRO: IOError - {fullJsonPath} - {e}")
        printHead()

    except TypeError as e:

        print(f"ERRO: Os dados fornecidos não são serializáveis para JSON - {e}")
        printHead()


def LoadDataFromJson(dirPath: str, jsonFileName: str) -> list[dict]:

    printHead()

    fullJsonPath = os.path.join(dirPath, jsonFileName)
    loadedData: list[dict] = [] 

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

getData()