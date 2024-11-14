#!/usr/bin/env python3
import time
import csv
import random
import numpy as np

# Definición de la clase NeuralNetwork
class NeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        self.num_layers  = len(layers)
        self.layer_sizes = layers
        self.biases =[np.random.randn(y,1) for y in layers[1:]] if biases == None else biases
        self.weights=[np.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])] if weights==None else weights
        
    def feedforward(self, x):
        for i in range(len(self.biases)):
            z = np.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + np.exp(-z))  # Sigmoid
        return x

    def feedforward_verbose(self, x):
        y = [x]
        for i in range(len(self.biases)):
            z = np.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + np.exp(-z))
            y.append(x)
        return y

    def backpropagate(self, x, yt):
        y = self.feedforward_verbose(x)
        delta = (y[-1] - yt) * y[-1] * (1 - y[-1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, y[-2].T)

        for l in range(2, self.num_layers):
            WT = self.weights[-l + 1].T
            delta = np.dot(WT, delta) * y[-l] * (1 - y[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, y[-l - 1].T)
        
        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        M = len(batch)
        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backpropagate(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - eta * nw / M for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - eta * nb / M for b, nb in zip(self.biases , nabla_b)]

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.update_with_batch(batch, eta)

# Función para cargar el conjunto de datos
def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [], [], [], []
    for i in range(10):
        f_data = [c / 255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [np.asarray(f_data[784 * j:784 * (j + 1)]).reshape([784, 1]) for j in range(1000)]
        label = np.asarray([1 if i == j else 0 for j in range(10)]).reshape([10, 1])
        training_dataset += images[0:len(images) // 2]
        training_labels += [label for j in range(len(images) // 2)]
        testing_dataset += images[len(images) // 2:len(images)]
        testing_labels += [label for j in range(len(images) // 2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

# Configuración para los experimentos
learning_rates = [0.5, 1.0, 3.0, 10.0]
epochs_list = [3, 10, 50, 100]
batch_sizes = [5, 10, 30, 100]
architectures = [
    [784, 15, 10],
    [784, 30, 10],
    [784, 50, 10],
    [784, 30, 30, 10]
]

# Cargar el conjunto de datos
training_dataset, testing_dataset = load_dataset("/home/oscar/Mobile-Robots-2025-1/catkin_ws/src/vision/neural_network/handwritten_digits/")

# Función para realizar el experimento con una configuración dada
def run_experiment(architecture, learning_rate, epochs, batch_size):
    nn = NeuralNetwork(architecture)    
    start_time = time.time()    
    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
    training_time = time.time() - start_time
    
    success_count = 0
    for _ in range(100):
        img, label = random.choice(testing_dataset)
        output = nn.feedforward(img)
        predicted_label = np.argmax(output)
        true_label = np.argmax(label)
        if predicted_label == true_label:
            success_count += 1
            
    accuracy = (success_count / 100) * 100    
    return training_time, accuracy

# Guardar resultados en un archivo CSV
output_file = "resultados_experimentos.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Architecture", "Learning Rate", "Epochs", "Batch Size", "Training Time (s)", "Accuracy (%)"])
    
    for architecture in architectures:
        for learning_rate in learning_rates:
            for epochs in epochs_list:
                for batch_size in batch_sizes:
                    print(f"Corriendo experimento - Arquitectura: {architecture}, Learning Rate: {learning_rate}, Epochs: {epochs}, Batch Size: {batch_size}")
                    training_time, accuracy = run_experiment(architecture, learning_rate, epochs, batch_size)
                    writer.writerow([architecture, learning_rate, epochs, batch_size, training_time, accuracy])

print(f"Experimentos completados. Resultados guardados en {output_file}.")

