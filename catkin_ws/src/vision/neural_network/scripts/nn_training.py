#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2024-2
# TRAINING A NEURAL NETWORK
#
# Instructions:
# Complete the code to train a neural network for
# handwritten digits recognition.
#
import cv2
import sys
import random
import numpy
import rospy
import rospkg
import csv
import os
import getpass
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


NAME = "Larios Avila Armando"

class NeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        #
        # The list 'layers' indicates the number of neurons in each layer.
        # Remember that the first layer indicates the dimension of the inputs and thus,
        # there is no bias vector fot the first layer.
        # For this practice, 'layers' should be something like [784, n2, n3, ..., nl, 10]
        # All weights and biases are initialized with random values. In each layer we have a matrix
        # of weights where row j contains all the weights of the j-th neuron in that layer. For this example,
        # the first matrix should be of order n2 x 784 and last matrix should be 10 x nl.
        #
        self.num_layers  = len(layers)
        self.layer_sizes = layers
        self.biases =[numpy.random.randn(y,1) for y in layers[1:]] if biases == None else biases
        self.weights=[numpy.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])] if weights==None else weights
        
    def feedforward(self, x):
        #
        # This function gets the output of the network when input is 'x'.
        #
        for i in range(len(self.biases)):
            z = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-z))  #output of the current layer is the input of the next one
        return x

    def feedforward_verbose(self, x):
        y = []
        #
        # TODO:
        # Write a function similar to 'feedforward' but instead of returning only the output layer,
        # return a list containing the output of each layer, from input to output.
        # Include input x as the first output.
        #
        y.append(x)
        for i in range(len(self.biases)):
            z = numpy.dot(self.weights[i], x) + self.biases[i]
            x = 1.0 / (1.0 + numpy.exp(-z))  #output of the current layer is the input of the next one
            y.append(x)
        
        return y

    def backpropagate(self, x, yt):
        y = self.feedforward_verbose(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        # TODO:
        # Return a tuple [nabla_w, nabla_b] containing the gradient of cost function C with respect to
        # each weight and bias of all the network. The gradient is calculated assuming only one training
        # example is given: the input 'x' and the corresponding label 'yt'.
        # nabla_w and nabla_b should have the same dimensions as the corresponding
        # self.weights and self.biases
        # You can calculate the gradient following these steps:
        #
        # Calculate delta for the output layer L: delta=(yL-yt)*yL*(1-yL)
        # nabla_b of output layer = delta      
        # nabla_w of output layer = delta*yLpT where yLpT is the transpose of the ouput vector of layer L-1
        # FOR all layers 'l' from L-1 to input layer: 
        #     delta = (WT * delta)*yl*(1 - yl)
        #     where 'WT' is the transpose of the matrix of weights of layer l+1 and 'yl' is the output of layer l
        #     nabla_b[-l] = delta
        #     nabla_w[-l] = delta*ylpT  where ylpT is the transpose of outputs vector of layer l-1
        #
        
        delta=(y[-1] - yt)*y[-1]*(1-y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = delta*y[-2].T
        for i in range (2,self.num_layers):
            delta = numpy.dot(self.weights[-i+1].T,delta)*y[-i]*(1 - y[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = delta*y[-i-1].T
        
        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        #
        # This function exectutes gradient descend for the subset of examples
        # given by 'batch' with learning rate 'eta'
        # 'batch' is a list of training examples [(x,y), ..., (x,y)]
        #
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        M = len(batch)
        for x,y in batch:
            if rospy.is_shutdown():
                break
            delta_nabla_w, delta_nabla_b = self.backpropagate(x,y)
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w-eta*nw/M for w,nw in zip(self.weights, nabla_w)]
        self.biases  = [b-eta*nb/M for b,nb in zip(self.biases , nabla_b)]
        return nabla_w, nabla_b

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum([numpy.sum(n) for n in [nw*nw for nw in nabla_w]])
        mag_b = sum([numpy.sum(b) for b in [nb*nb for nb in nabla_b]])
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0,len(training_data), batch_size)]
            for batch in batches:
                if rospy.is_shutdown():
                    return
                nabla_w, nabla_b = self.update_with_batch(batch, eta)
                sys.stdout.write("\rGradient magnitude: %f            " % (self.get_gradient_mag(nabla_w, nabla_b)))
                sys.stdout.flush()
            print("Epoch: " + str(j))
    #
    ### END OF CLASS
    #


def load_dataset(folder):
    print("Loading data set from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [],[],[],[]
    for i in range(10):
        f_data = [c/255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784*j:784*(j+1)]).reshape([784,1]) for j in range(1000)]
        label  = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10,1])
        training_dataset += images[0:len(images)//2]
        training_labels  += [label for j in range(len(images)//2)]
        testing_dataset  += images[len(images)//2:len(images)]
        testing_labels   += [label for j in range(len(images)//2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

def generar_graficas(output_folder, csv_file_path):
    # Leer los datos del archivo CSV
    data = {
        "learning_rate": [],
        "epochs": [],
        "batch_size": [],
        "training_time": [],
        "success_rate": []
    }

    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data["learning_rate"].append(float(row["Learning Rate"]))
            data["epochs"].append(int(row["Epochs"]))
            data["batch_size"].append(int(row["Batch Size"]))
            data["training_time"].append(float(row["Training Time (ms)"]))
            data["success_rate"].append(float(row["Success Rate (%)"]))

    # Lista de parámetros y etiquetas para iterar en la creación de gráficas
    parametros = ["learning_rate", "epochs", "batch_size"]
    etiquetas = ["Learning Rate", "Epochs", "Batch Size"]

    # Generar y guardar las gráficas
    for i, parametro in enumerate(parametros):
        # Gráfica del parámetro vs Tiempo de Entrenamiento
        plt.figure()
        plt.scatter(data[parametro], data["training_time"], color="blue")
        plt.xlabel(etiquetas[i])
        plt.ylabel("Training Time (ms)")
        plt.title(f"{etiquetas[i]} vs Training Time")
        plt.grid()
        training_time_path = os.path.join(output_folder, f"{etiquetas[i]}_vs_Training_Time.png")
        plt.savefig(training_time_path)
        plt.close()
        print(f"Gráfica guardada en: {training_time_path}")

        # Gráfica del parámetro vs Porcentaje de Éxitos
        plt.figure()
        plt.scatter(data[parametro], data["success_rate"], color="green")
        plt.xlabel(etiquetas[i])
        plt.ylabel("Success Rate (%)")
        plt.title(f"{etiquetas[i]} vs Success Rate")
        plt.grid()
        success_rate_path = os.path.join(output_folder, f"{etiquetas[i]}_vs_Success_Rate.png")
        plt.savefig(success_rate_path)
        plt.close()
        print(f"Gráfica guardada en: {success_rate_path}")


def main():
    print("TRAINING A NEURAL NETWORK - " + NAME)
    rospy.init_node("nn_training")
    rospack = rospkg.RosPack()
    dataset_folder = rospack.get_path("neural_network") + "/handwritten_digits/"
    
    # Ruta de la carpeta "RESULTADOS" en el escritorio
    username = getpass.getuser()
    desktop_folder = f"/home/{username}/Escritorio/"
    if not os.path.exists(desktop_folder):
        desktop_folder = f"/home/{username}/Desktop/"
    
    output_folder = os.path.join(desktop_folder, "RESULTADOS")
    os.makedirs(output_folder, exist_ok=True)

    # Archivo CSV único para almacenar todos los resultados
    csv_file_path = os.path.join(output_folder, "resultados_entrenamiento.csv")

    cmd = 0
    
    if rospy.has_param("~epochs"):
        epochs = rospy.get_param("~epochs")
    if rospy.has_param("~batch_size"):
        batch_size = rospy.get_param("~batch_size")
    if rospy.has_param("~learning_rate"):
        learning_rate = rospy.get_param("~learning_rate") 

    training_dataset, testing_dataset = load_dataset(dataset_folder)
    
    try:
        saved_data = numpy.load(dataset_folder + "network.npz", allow_pickle=True)
        layers = [saved_data['w'][0].shape[1]] + [b.shape[0] for b in saved_data['b']]
        nn = NeuralNetwork(layers, weights=saved_data['w'], biases=saved_data['b'])
        print("Loading data from previously trained model with layers " + str(layers))
    except:
        nn = NeuralNetwork([784, 30, 10])
        pass
    
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Learning Rate", "Epochs", "Batch Size", "Training Time (ms)", "Success Rate (%)"])

    # Iterar sobre todas las combinaciones de parámetros
    for learning_rate in [0.5, 1.0, 3.0, 10.0]:
        for epochs in [3, 10, 50, 100]:
            for batch_size in [5, 10, 30, 100]:
                print(f"Tasa de aprendizaje = {learning_rate}, Épocas = {epochs}, Tamaño de lote = {batch_size}")
                
                # Calcular tiempo de entrenamiento
                start_time = rospy.Time.now()
                nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
                end_time = rospy.Time.now()
                training_time = 1000 * (end_time - start_time).to_sec()

                hits, nohits = 0, 0
                true_labels = []  # Lista para etiquetas verdaderas
                predicted_labels = []  # Lista para etiquetas predichas

                # Realizar 100 iteraciones de prueba
                for i in range(100):
                    img, label = testing_dataset[numpy.random.randint(0, 4999)]
                    y = nn.feedforward(img).transpose()
                    
                    expected = numpy.argmax(label.transpose())
                    recognized = numpy.argmax(y)

                    # Almacenar etiquetas para la matriz de confusión
                    true_labels.append(expected)
                    predicted_labels.append(recognized)
                    
                    if expected == recognized:
                        hits += 1
                    else:
                        nohits += 1

                # Calcular porcentaje de éxitos
                success_rate = (hits / 100) * 100
                print(f"Tiempo de entrenamiento: {training_time} ms, Porcentaje de éxitos: {success_rate}%")

                # Generar matriz de confusión
                conf_matrix = confusion_matrix(true_labels, predicted_labels)
                print(f"Matriz de confusión para LR={learning_rate}, Epochs={epochs}, Batch Size={batch_size}:\n{conf_matrix}")

                # Guardar los resultados en el archivo CSV único
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([learning_rate, epochs, batch_size, training_time, success_rate])

                # Guardar la matriz de confusión en un archivo CSV separado si se desea
                conf_matrix_path = os.path.join(output_folder, f"conf_matrix_LR{learning_rate}_Epochs{epochs}_Batch{batch_size}.csv")
                #numpy.savetxt(conf_matrix_path, conf_matrix, delimiter=",", fmt='%d')
                with open(conf_matrix_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    
                    # Crear encabezado para las columnas de "Expected"
                    headers = ["Recognized \\ Expected"] + [str(i) for i in range(len(conf_matrix))]
                    writer.writerow(headers)

                    # Escribir cada fila de la matriz de confusión con la estructura solicitada
                    for i, row in enumerate(conf_matrix):
                        writer.writerow([i] + list(row))
                
                print(f"Matriz de confusión guardada en: {conf_matrix_path}")

    # Generar gráficas al finalizar el programa
    generar_graficas(output_folder, csv_file_path)
    print(f"Resultados guardados en: {csv_file_path}")


if __name__ == '__main__':
    main()