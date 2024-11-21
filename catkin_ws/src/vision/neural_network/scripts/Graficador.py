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
        56, 76, 84, 76, 88, 88, 91, 85, 91, 85, 89, 91, 89, 91, 92, 86,
        91, 89, 85, 91, 86, 91, 93, 89, 92, 87, 92, 88, 86, 93, 87, 89,
        91, 91, 93, 87, 89, 91, 81, 93, 94, 95, 96, 90, 89, 89, 91, 91,
        89, 86, 94, 95, 96, 94, 93, 90, 93, 95, 91, 88, 92, 95, 94, 90
    ],
    'Tiempo_entrenamiento': [
        1.199202, 1.138190, 1.145716, 1.154287, 0.988543, 0.961847, 0.939444, 0.932256,
        0.799349, 0.804358, 0.800748, 0.792012, 0.776229, 0.766793, 0.759087, 0.758006,
        3.804063, 3.807302, 3.834759, 3.786610, 3.182657, 3.158604, 3.154153, 3.450497,
        2.798583, 2.952808, 3.232888, 2.993392, 3.326494, 3.026224, 2.938882, 2.910001,
        20.164782, 20.695940, 20.765705, 21.162712, 17.429817, 17.060049, 16.760306, 17.396563,
        15.152079, 15.646001, 14.721789, 14.866364, 14.313744, 14.247896, 17.475863, 13.893701,
        41.203193, 40.071878, 41.589485, 41.735944, 32.683028, 32.505779, 32.642916, 32.591262,
        29.065448, 30.038376, 30.203268, 28.606099, 26.668354, 26.590598, 28.012031, 27.728795
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

