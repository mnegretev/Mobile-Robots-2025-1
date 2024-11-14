import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Datos proporcionados
learning_rates = [0, 1, 3, 10] * 16
batch_sizes = [5, 10, 30, 100] * 16
epochs = [3] * 16 + [10] * 16 + [50] * 16 + [100] * 16
training_times = [
    3.188976, 2.809508, 2.839136, 2.697013, 2.148355, 1.783186, 2.439831, 2.891252, 
    1.690798, 1.532811, 1.580012, 1.553713, 1.488470, 1.462628, 1.502037, 1.659017, 
    8.043034, 11.551887, 8.426347, 9.949498, 7.072607, 6.369263, 9.313341, 12.211608, 
    9.817968, 6.741392, 10.220356, 10.075306, 7.009040, 9.732899, 6.511467, 6.661159, 
    73.735548, 47.209211, 42.531879, 38.641904, 43.407454, 53.456774, 40.894847, 39.701984, 
    29.240318, 70.277860, 34.831421, 30.649472, 30.615019, 28.519919, 38.908900, 30.007697, 
    89.737738, 97.401321, 97.353635, 93.905095, 68.735041, 71.253349, 69.807605, 88.929274, 
    58.835302, 57.238415, 59.645619, 103.413233, 61.196114, 66.300363, 61.174970, 68.813983
]
accuracies = [
    59, 90, 84, 84, 91, 85, 86, 90, 92, 86, 95, 90, 89, 88, 89, 88, 
    90, 92, 87, 91, 91, 87, 92, 88, 91, 93, 93, 90, 95, 93, 91, 95, 
    98, 94, 86, 87, 92, 87, 92, 95, 92, 95, 87, 94, 90, 87, 93, 88, 
    97, 87, 92, 93, 92, 88, 93, 84, 92, 95, 90, 85, 90, 91, 96, 92
]

# Crear un DataFrame con los datos para facilitar el análisis
data = pd.DataFrame({
    "Learning Rate": learning_rates,
    "Batch Size": batch_sizes,
    "Epochs": epochs,
    "Training Time (s)": training_times,
    "Accuracy": accuracies
})

# Crear la figura para las tres gráficas adicionales
plt.figure(figsize=(16, 12))

# Gráfico 1: Línea de Epochs vs Training Time, agrupado por Batch Size y Learning Rate
plt.subplot(3, 1, 1)
sns.lineplot(data=data, x="Epochs", y="Training Time (s)", hue="Batch Size", style="Learning Rate", markers=True, dashes=False)
plt.title("Tiempo de Entrenamiento por Configuración de Epochs y Batch Size")

# Gráfico 2: Línea de Epochs vs Accuracy, agrupado por Batch Size y Learning Rate
plt.subplot(3, 1, 2)
sns.lineplot(data=data, x="Epochs", y="Accuracy", hue="Batch Size", style="Learning Rate", markers=True, dashes=False)
plt.title("Número de Aciertos por Configuración de Epochs y Batch Size")

# Gráfico 3: Mapa de calor de Número de Aciertos por Batch Size y Learning Rate
plt.subplot(3, 1, 3)
heatmap_data = data.pivot_table(values="Accuracy", index="Batch Size", columns="Learning Rate", aggfunc="mean")
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", cbar_kws={'label': 'Accuracy'})
plt.title("Mapa de Calor del Número de Aciertos por Batch Size y Learning Rate")

# Ajustar el espaciado entre las subgráficas
plt.tight_layout()

# Mostrar las gráficas
plt.show()
