import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import LambdaCallback, EarlyStopping
from keras.optimizers import Adam

# Ruta a la carpeta con imágenes
image_folder = "/home/salazaar01/Mobile-Robots-2025-1/catkin_ws/src/vision/neural_network/handwritten_digits/Fruits_Classification/Images"

# Lista donde se almacenarán las imágenes cargadas
images = []

# Lista donde se almacenarán las etiquetas
labels = []

# Cargar todas las imágenes de la carpeta
for filename in os.listdir(image_folder):
    filepath = os.path.join(image_folder, filename)
    
    # Verificar si el archivo es una imagen
    if filename.endswith(('.jpg', '.png', 'jpeg')):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            images.append(image)
            
            # Asignar la etiqueta basada en el nombre del archivo
            filename_lower = filename.lower()  # Convertir el nombre del archivo a minúsculas para hacer la comparación más robusta
            if 'grape' in filename_lower:
                labels.append(0)  # Grapes -> 0
            elif 'banana' in filename_lower:
                labels.append(1)  # Banana -> 1
            elif 'apple' in filename_lower:
                labels.append(2)  # Apple -> 2
            else:
                print(f"Etiqueta desconocida en {filename}")
        else:
            print(f"No se pudo cargar la imagen: {filename}")

# Verificar el total de imágenes y etiquetas cargadas
print(f"Total de imágenes cargadas: {len(images)}")
print(f"Total de etiquetas asignadas: {len(labels)}")

# Asegurarse de que el número de imágenes y etiquetas coincidan
assert len(images) == len(labels), "El número de imágenes y etiquetas no coincide"

# Función de preprocesamiento (redimensionamiento y normalización)
def preprocess_dataset(dataset, target_size=(128, 128)):
    """
    Redimensiona y normaliza todas las imágenes de un dataset.
    
    :param dataset: Lista o arreglo de imágenes (en formato NumPy o similar).
    :param target_size: Tamaño objetivo (ancho, alto) para las imágenes.
    :return: Dataset procesado como un arreglo de NumPy.
    """
    processed_images = []
    for idx, image in enumerate(dataset):
        # Redimensionar la imagen
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        # Normalizar los valores de píxeles a [0, 1]
        normalized_image = resized_image / 255.0
        # Asegurarse de que la imagen tenga un solo canal (escala de grises)
        image_with_channel = np.expand_dims(normalized_image, axis=-1)  # Agregar la dimensión de canal
        processed_images.append(image_with_channel)
        
        print(f"Imagen {idx} redimensionada a {target_size} y normalizada")
    
    # Convertir la lista a un arreglo de NumPy
    return np.array(processed_images)

# Preprocesar el dataset (redimensionar y normalizar)
processed_dataset = preprocess_dataset(images)

# Convertir las etiquetas a one-hot encoding
labels_one_hot = to_categorical(labels, num_classes=3)

# Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(processed_dataset, labels_one_hot, test_size=0.2, random_state=42)

# Verificar el tamaño del dataset procesado
print(f"Total de imágenes procesadas: {processed_dataset.shape[0]}")

# Visualizar las primeras 5 imágenes con sus etiquetas
for idx in range(5):  # Mostrar las primeras 5 imágenes
    image_to_display = processed_dataset[idx].reshape(128, 128)  # Reshape para visualizar como imagen 2D
    plt.imshow(image_to_display, cmap='gray')
    plt.title(f"Etiqueta: {labels[idx]} - {os.listdir(image_folder)[idx]}")
    plt.show()

# Red neuronal (utilizando el modelo de red convolucional con mejoras)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))  # Capa convolucional
model.add(MaxPooling2D((2, 2)))  # Capa de pooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())  # Aplanar la imagen para las capas densas
model.add(Dense(256, activation='relu'))  # Capa oculta más grande
model.add(Dropout(0.5))  # Regularización con Dropout
model.add(Dense(128, activation='relu'))  # Capa oculta
model.add(Dense(3, activation='softmax'))  # Capa de salida con 3 neuronas (una por clase)

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Función para visualizar las predicciones después de cada época
def visualize_predictions(epoch, logs):
    # Mostrar imágenes solo después de la 10ª época
    if epoch >= 10:
        # Obtener un subconjunto aleatorio de imágenes a predecir (por ejemplo, 5 imágenes aleatorias del conjunto de prueba)
        random_indices = np.random.choice(len(X_test), 5, replace=False)  # Seleccionar 5 índices aleatorios sin repetición
        predictions = model.predict(X_test[random_indices])  # Predicciones sobre las imágenes seleccionadas
        predicted_labels = np.argmax(predictions, axis=1)  # Etiquetas predichas

        # Mapeo de etiquetas a nombres
        class_names = {0: 'Grape', 1: 'Banana', 2: 'Apple'}

        # Visualizar las imágenes seleccionadas con su predicción y etiqueta real
        for idx in range(50):  # Mostrar las 5 imágenes seleccionadas
            image_to_display = X_test[random_indices[idx]].reshape(128, 128)  # Reshape para visualizar como imagen 2D
            predicted_label = predicted_labels[idx]  # Etiqueta predicha
            real_label = np.argmax(y_test[random_indices[idx]])  # Etiqueta real
            
            plt.imshow(image_to_display, cmap='gray')
            plt.title(f"Predicción: {class_names[predicted_label]} - Real: {class_names[real_label]}")  # Mostrar nombre de la clase
            plt.show()

# Definir el callback para visualizar las predicciones
visualize_callback = LambdaCallback(on_epoch_end=visualize_predictions)

# Early stopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo con el callback
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[visualize_callback, early_stopping])

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%")

