import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Datos reales proporcionados
epochs = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 100, 100, 100, 100, 
          100, 100, 100, 100, 100, 100, 100]
batch_sizes = [5, 5, 5, 5, 10, 10, 10, 10, 30, 30, 30, 30, 100, 100, 100, 100, 5, 5, 5, 5, 10, 10, 10, 10, 30, 30, 30, 30,
               100, 100, 100, 100, 5, 5, 5, 5, 10, 10, 10, 10, 30, 30, 30, 30, 100, 100, 100, 100, 5, 5, 5, 5, 10, 10, 10, 
               10, 30, 30, 30, 30, 100, 100, 100, 100]
learning_rates = [0, 1, 3, 10] * 16
training_times = [3.188976, 2.809508, 2.839136, 2.697013, 2.148355, 1.783186, 2.439831, 2.891252, 1.690798, 1.532811, 
                  1.580012, 1.553713, 1.488470, 1.462628, 1.502037, 1.659017, 8.043034, 11.551887, 8.426347, 9.949498, 
                  7.072607, 6.369263, 9.313341, 12.211608, 9.817968, 6.741392, 10.220356, 10.075306, 7.009040, 9.732899, 
                  6.511467, 6.661159, 73.735548, 47.209211, 42.531879, 38.641904, 43.407454, 53.456774, 40.894847, 
                  39.701984, 29.240318, 70.277860, 34.831421, 30.649472, 30.615019, 28.519919, 38.908900, 30.007697, 
                  89.737738, 97.401321, 97.353635, 93.905095, 68.735041, 71.253349, 69.807605, 88.929274, 58.835302, 
                  57.238415, 59.645619, 103.413233, 61.196114, 66.300363, 61.174970, 68.813983]
accuracies = [59, 90, 84, 84, 91, 85, 86, 90, 92, 86, 95, 90, 89, 88, 89, 88, 90, 92, 87, 91, 91, 87, 92, 88, 91, 93, 93, 
              90, 95, 93, 91, 95, 98, 94, 86, 87, 92, 87, 92, 95, 92, 95, 87, 94, 90, 87, 93, 88, 97, 87, 92, 93, 92, 88, 
              93, 84, 92, 95, 90, 85, 90, 91, 96, 92]

# Crear la figura con un tamaño más grande y espaciado adecuado
plt.figure(figsize=(14, 12))

# Gráfica 1: Batch Size vs Training Time
plt.subplot(3, 2, 1)
plt.scatter(batch_sizes, training_times, color='blue')
plt.xlabel('Batch Size')
plt.ylabel('Training Time (s)')
plt.title('Batch Size vs Training Time')

# Gráfica 2: Batch Size vs Accuracy
plt.subplot(3, 2, 2)
plt.scatter(batch_sizes, accuracies, color='red')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title('Batch Size vs Accuracy')

# Gráfica 3: Learning Rate vs Training Time
plt.subplot(3, 2, 3)
plt.scatter(learning_rates, training_times, color='green')
plt.xlabel('Learning Rate')
plt.ylabel('Training Time (s)')
plt.title('Learning Rate vs Training Time')

# Gráfica 4: Learning Rate vs Accuracy
plt.subplot(3, 2, 4)
plt.scatter(learning_rates, accuracies, color='purple')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Learning Rate vs Accuracy')

# Gráfica 5: Epochs vs Training Time
plt.subplot(3, 2, 5)
plt.scatter(epochs, training_times, color='orange')
plt.xlabel('Epochs')
plt.ylabel('Training Time (s)')
plt.title('Epochs vs Training Time')

# Gráfica 6: Epochs vs Accuracy
plt.subplot(3, 2, 6)
plt.scatter(epochs, accuracies, color='brown')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Epochs vs Accuracy')

# Ajustar el espaciado entre subgráficas
plt.tight_layout()
plt.show()

# Agregar las nuevas gráficas
# Gráfica 7: Tiempo de Entrenamiento por Configuración de Epochs y Batch Size
plt.figure(figsize=(8, 6))
for lr in sorted(set(learning_rates)):
    for bs in sorted(set(batch_sizes)):
        times = [training_times[i] for i in range(len(training_times)) if learning_rates[i] == lr and batch_sizes[i] == bs]
        epchs = [epochs[i] for i in range(len(epochs)) if learning_rates[i] == lr and batch_sizes[i] == bs]
        plt.plot(epchs, times, label=f'Batch Size: {bs}, LR: {lr}', linestyle='--' if lr == 10 else '-')
plt.xlabel('Epochs')
plt.ylabel('Training Time (s)')
plt.title('Tiempo de Entrenamiento por Configuración de Epochs y Batch Size')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Gráfica 8: Número de Aciertos por Configuración de Epochs y Batch Size
plt.figure(figsize=(8, 6))
for lr in sorted(set(learning_rates)):
    for bs in sorted(set(batch_sizes)):
        accs = [accuracies[i] for i in range(len(accuracies)) if learning_rates[i] == lr and batch_sizes[i] == bs]
        epchs = [epochs[i] for i in range(len(epochs)) if learning_rates[i] == lr and batch_sizes[i] == bs]
        plt.plot(epchs, accs, label=f'Batch Size: {bs}, LR: {lr}', linestyle='--' if lr == 10 else '-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Número de Aciertos por Configuración de Epochs y Batch Size')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# Gráfica 9: Mapa de Calor del Número de Aciertos por Batch Size y Learning Rate
# Preparar los datos para el mapa de calor
heatmap_data = np.zeros((len(set(batch_sizes)), len(set(learning_rates))))
for i, bs in enumerate(sorted(set(batch_sizes))):
    for j, lr in enumerate(sorted(set(learning_rates))):
        acc = [accuracies[k] for k in range(len(accuracies)) if batch_sizes[k] == bs and learning_rates[k] == lr]
        heatmap_data[i, j] = np.mean(acc) if acc else np.nan

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", xticklabels=sorted(set(learning_rates)), yticklabels=sorted(set(batch_sizes)), cmap="coolwarm")
plt.xlabel('Learning Rate')
plt.ylabel('Batch Size')
plt.title('Mapa de Calor del Número de Aciertos por Batch Size y Learning Rate')
plt.show()

