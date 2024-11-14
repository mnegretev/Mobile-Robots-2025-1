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
        57, 70, 77, 63, 85, 83, 88, 88, 91, 85, 91, 89, 87, 90, 92, 92,
        92, 90, 89, 86, 92, 84, 91, 92, 89, 93, 91, 92, 94, 92, 88, 91,
        94, 90, 87, 95, 94, 91, 91, 94, 94, 93, 91, 90, 93, 91, 89, 93,
        92, 93, 96, 85, 90, 89, 88, 93, 90, 94, 93, 90, 90, 89, 88, 89
    ],
    'Tiempo_entrenamiento': [
        1.165498, 1.047921, 1.044923, 1.051233, 0.887274, 0.879838, 0.882511, 0.893612,
        0.749102, 0.759363, 0.795019, 0.806748, 0.734999, 0.723638, 0.784096, 0.779602,
        3.622634, 3.551485, 3.524653, 3.561257, 3.150481, 2.981141, 2.966516, 2.977225,
        2.608617, 2.544825, 2.510767, 2.520990, 2.412964, 2.428416, 2.378435, 2.400073,
        18.187429, 17.742125, 18.313378, 18.295355, 15.277247, 15.216470, 14.851803, 14.902763,
        12.801796, 12.686044, 13.255965, 13.251273, 12.098871, 12.025609, 12.694433, 12.305721,
        36.203565, 35.323725, 37.102643, 35.043371, 29.610605, 29.614791, 29.592042, 29.742660,
        25.636409, 25.361143, 25.286377, 25.140604, 23.987768, 24.110188, 24.106865, 24.187315
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

