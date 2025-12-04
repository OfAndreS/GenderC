import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import GroupShuffleSplit

from src.Shared import utils

def PrepareDatasets(testSize, validationSize):

    xList, yList, zMapping, groupsList = utils.getData()

    x = np.array(xList)
    y = np.array(yList)
    groups = np.array(groupsList)

    gss = GroupShuffleSplit(n_splits=1, test_size=testSize, random_state=42)
    trainIdx, testIdx = next(gss.split(x, y, groups))

    xTrain, xTest = x[trainIdx], x[testIdx]
    yTrain, yTest = y[trainIdx], y[testIdx]
    groupsTrain = groups[trainIdx] 

    gssVal = GroupShuffleSplit(n_splits=1, test_size=validationSize, random_state=42)
    tIdx, vIdx = next(gssVal.split(xTrain, yTrain, groupsTrain))

    xTrain, xValidation = xTrain[tIdx], xTrain[vIdx]
    yTrain, yValidation = yTrain[tIdx], yTrain[vIdx]

    xTrain = xTrain[..., np.newaxis]
    xValidation = xValidation[..., np.newaxis]
    xTest = xTest[..., np.newaxis]

    return xTrain, xValidation, xTest, yTrain, yValidation, yTest


def BuildModelOptimized(inputShape):

    model = keras.Sequential()
    
    reg = keras.regularizers.l2(0.001)
    init = 'he_normal' 

    # 1ª Camada
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape, kernel_regularizer=reg, padding='same', kernel_initializer=init))
    model.add(keras.layers.MaxPooling2D((2, 2), padding='same')) 
    model.add(keras.layers.BatchNormalization())

    # 2ª Camada
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=reg, padding='same', kernel_initializer=init)) 
    model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2)) 

    # 3ª Camada
    model.add(keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_regularizer=reg, padding='same', kernel_initializer=init)) 
    model.add(keras.layers.MaxPooling2D((2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten()) 
    
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=reg, kernel_initializer=init)) 
    model.add(keras.layers.Dropout(0.4)) 

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def TrainModel(model, xTrain, yTrain, xValidation, yValidation):

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(xTrain, yTrain,
                        validation_data=(xValidation, yValidation),
                        batch_size=32,
                        epochs=30)
    return history


def Predict(model, x, y):

    x = x[np.newaxis, ...] 

    prediction = model.predict(x)

    predictedIndex = np.argmax(prediction, axis=1)

    print(f"Expected index: {y}, Predicted index: {predictedIndex[0]}")


if __name__ == "__main__":
    
    xTrain, xValidation, xTest, yTrain, yValidation, yTest = PrepareDatasets(0.25, 0.2)

    inputShape = (xTrain.shape[1], xTrain.shape[2], 1)
    model = BuildModelOptimized(inputShape)

    history = TrainModel(model, xTrain, yTrain, xValidation, yValidation)
    
    testLoss, testAcc = model.evaluate(xTest, yTest, verbose=2)
    print(f'\nTest accuracy: {testAcc}')

    utils.SaveDataInJson(history, "output", "h.json")
    model.save("models/music_genre_cnn_valerio.h5")
    
    xToPredict = xTest[100]
    yToPredict = yTest[100]
    Predict(model, xToPredict, yToPredict)