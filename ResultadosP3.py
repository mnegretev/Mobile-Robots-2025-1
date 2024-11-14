import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define los datos
data = {
    'Prueba': range(1, 65),
    'Epochs': [3]*16 + [10]*16 + [50]*16 + [100]*16,
    'Batch_size': [5, 5, 5, 5, 10, 10, 10, 10, 30, 30, 30, 30, 100, 100, 100, 100]*4,
    'Learning_rate': [0, 1, 3, 10]*16,
    'Numero_aciertos': [
        53, 84, 86, 90, 90, 90, 92, 85, 92, 87, 91, 90, 93, 90, 92, 93,
        89, 92, 96, 86, 91, 87, 90, 92, 91, 89, 93, 90, 91, 87, 92, 91,
        93, 92, 92, 86, 93, 92, 89, 95, 93, 90, 94, 89, 94, 92, 93, 93,
        90, 97, 95, 91, 89, 94, 91, 92, 95, 92, 94, 90, 94, 93, 87, 92
    ],
    'Tiempo_entrenamiento': [
        2.742505, 2.194708, 2.571817, 2.723307, 2.484103, 2.277029, 2.217988, 2.034207,
        1.677487, 1.666925, 1.659004, 1.688703, 1.608845, 1.551706, 1.462955, 1.464458,
        8.415014, 7.792630, 8.517017, 7.912844, 11.030959, 6.661025, 6.537812, 7.923514,
        6.192724, 6.120900, 6.273304, 5.979798, 4.633411, 4.439193, 4.478259, 5.029748,
        43.955939, 44.379240, 52.355233, 43.379038, 27.094054, 27.950182, 30.879438, 29.066083,
        27.088037, 24.842656, 24.949550, 25.270984, 21.334825, 21.197592, 22.975257, 22.884279,
        74.886564, 72.112488, 73.209761, 72.700435, 60.188819, 62.020434, 62.411458, 61.617802,
        55.924206, 60.010581, 53.482322, 54.497961, 55.235369, 54.434048, 45.763186, 45.363524
    ]
}

df = pd.DataFrame(data)

# 2. Gráfica de Número de Aciertos por Configuración de Epochs
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Epochs", y="Numero_aciertos", hue="Batch_size", style="Learning_rate", markers=True)
plt.title("Número de Aciertos por Configuración de Epochs y Batch Size")
plt.xlabel("Epochs")
plt.ylabel("Número de Aciertos")
plt.legend(title="Batch Size / Learning Rate")
plt.grid()
plt.show()

# 3. Gráfica de Tiempo de Entrenamiento por Configuración de Epochs
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Epochs", y="Tiempo_entrenamiento", hue="Batch_size", style="Learning_rate", markers=True)
plt.title("Tiempo de Entrenamiento por Configuración de Epochs y Batch Size")
plt.xlabel("Epochs")
plt.ylabel("Tiempo de Entrenamiento (s)")
plt.legend(title="Batch Size / Learning Rate")
plt.grid()
plt.show()

# 4. Mapa de Calor de Número de Aciertos por Batch Size y Learning Rate
pivot_aciertos = df.pivot_table(values="Numero_aciertos", index="Batch_size", columns="Learning_rate", aggfunc="mean")
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_aciertos, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Mapa de Calor del Número de Aciertos por Batch Size y Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Batch Size")
plt.show()

# 5. Nueva Gráfica: Mapa de Calor de Número de Aciertos por Epochs y Learning Rate
pivot_aciertos_epoch_lr = df.pivot_table(values="Numero_aciertos", index="Epochs", columns="Learning_rate", aggfunc="mean")
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_aciertos_epoch_lr, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Mapa de Calor del Número de Aciertos por Epochs y Learning Rate")
plt.xlabel("Learning Rate")
plt.ylabel("Epochs")
plt.show()

