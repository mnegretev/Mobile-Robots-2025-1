import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define los datos
data = {
    'Prueba': range(1, 126),
    'Epochs': [3]*25 + [10]*25 + [50]*25 + [75]*25 + [100]*25,
    'Batch_size': [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100]*5,
    'Learning_rate': [0, 1, 3, 5, 10]*25,
    "Numero_aciertos" : [
        396, 427, 435, 450, 423, 436, 454, 447, 446, 442,
        452, 457, 463, 463, 461, 466, 462, 451, 462, 456,
        468, 455, 455, 457, 457, 448, 451, 452, 457, 450,
        447, 437, 452, 459, 462, 448, 471, 453, 457, 467,
        463, 455, 459, 467, 464, 464, 462, 454, 457, 454,
        458, 450, 461, 462, 451, 451, 453, 472, 463, 463,
        461, 439, 449, 463, 451, 450, 455, 458, 446, 446,
        446, 447, 461, 454, 458, 451, 454, 456, 470, 460, 
        458, 474, 459, 465, 453, 449, 459, 467, 467, 456, 
        465, 472, 446, 461, 462, 462, 456, 456, 451, 461, 
        465, 458, 449, 463, 459, 450, 457, 452, 459, 458, 
        453, 454, 449, 451, 455, 460, 459, 450, 462, 457, 
        448, 455, 453, 453, 440         
        
    ],
    "Tiempo_entrenamiento" : [
    1.161558, 1.105185, 1.084552, 1.103158, 1.102206, 0.946802, 0.945989, 0.948944, 0.949104, 0.947344,
    0.841983, 0.833604, 0.828227, 0.820200, 0.811335, 0.816173, 0.791665, 0.811141, 0.804212, 0.801903,
    0.787527, 0.790750, 0.783713, 0.757837, 0.774337, 3.616960, 3.673541, 3.775292, 3.752603, 3.811705,
    3.318376, 3.184262, 3.369135, 3.255012, 3.174258, 2.795057, 2.813537, 2.889976, 2.769258, 2.753252,
    2.751127, 2.802753, 2.932885, 2.805440, 2.891866, 2.699833, 2.548517, 2.584536, 2.617924, 2.597605,
    18.162524, 17.824339, 17.733807, 18.321767, 18.711164, 16.951931, 16.378193, 15.709582, 16.516111, 16.559790,
    14.331164, 14.006113, 14.144218, 13.740855, 13.956988, 13.785002, 13.346759, 13.292258, 13.110978, 13.354430,
    13.111626, 13.016701, 12.914975, 12.890882, 12.994179, 27.128303, 27.169071, 27.043891, 27.179219, 27.107410,
    23.513930, 23.557219, 23.573866, 23.642718, 23.814870, 20.167880, 21.013597, 22.792641, 21.906806, 20.973898,
    20.253844, 20.287348, 20.395098, 20.340238, 20.246347, 20.072329, 20.211179, 19.451769, 19.976991, 19.561578,
    36.042694, 35.941928, 36.351300, 36.087604, 35.923386, 30.791343, 31.131600, 31.293630, 31.471428, 31.622892,
    27.576940, 27.578481, 27.599616, 27.401629, 27.509842, 26.592357, 26.724214, 26.627380, 26.591581, 27.426625,
    25.791643, 25.956863, 25.845157, 25.733057, 25.765876
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

