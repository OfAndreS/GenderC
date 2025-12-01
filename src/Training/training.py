import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

# Importa o módulo de utilitários partilhado
from src.Shared import utils


BATCH_SIZE = 32
EPOCHS = 15


def plot_history(history):
    """
    Gera gráficos de precisão (accuracy) e erro (loss)
    """
    fig, axs = plt.subplots(2)

    # Gráfico de Accuracy
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # Gráfico de Error/Loss
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    # Ajusta o layout e mostra (ou salva)
    plt.tight_layout()
    plt.savefig("graphics/g.png")


def prepare_datasets(test_size, validation_size):

    x_list, y_list, z_mapping = utils.getData()

    X = np.array(x_list)
    y = np.array(y_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """
    Gera o modelo CNN com técnicas para combater Overfitting (Regularização e Dropout)
    """

    model = keras.Sequential()

    # 1ª camada convolucional
    # Adicionado kernel_regularizer L2
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # Adicionado Dropout para evitar co-adaptação nas features iniciais
    model.add(keras.layers.Dropout(0.2))

    # 2ª camada convolucional
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # 3ª camada convolucional
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    # Camada densa (Flatten)
    model.add(keras.layers.Flatten())
    
    # Adicionado L2 na camada densa também
    model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    
    # Aumentado o Dropout para 0.5 (Drástico, mas necessário para overfitting pesado)
    model.add(keras.layers.Dropout(0.5))

    # Camada de saída
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def ExecuteModel():
    
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    
    model = build_model(input_shape)

    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train, y_train, 
                        validation_data=(X_validation, y_validation), 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS)
    
    plot_history(history)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')

    model.save("models/music_genre_cnn.h5")
    print("\nModelo salvo em models/music_genre_cnn.h5")

    return history

if __name__ == "__main__":
    ExecuteModel()