#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-1
# PATH SMOOTHING BY GRADIENT DESCENT
#
# Instructions:
# Write the code necessary to smooth a path using the gradient descent algorithm.
# MODIFY ONLY THE SECTIONS MARKED WITH THE 'TODO' COMMENT
#

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, PoseStamped, Point
from navig_msgs.srv import ProcessPath
from navig_msgs.srv import ProcessPathResponse

NAME = "LARIOS AVILA ARMANDO"

def smooth_path(Q, alpha, beta, max_steps):
    # Copiar el camino original
    P = np.copy(Q)
    tol = 0.00001  # Tolerancia para detener el algoritmo
    nabla = np.zeros(Q.shape)  # Gradientes inicializados en 0
    steps = 0  # Contador de pasos
    epsilon = 0.1  # Paso de ajuste del camino

    while steps < max_steps:
        max_delta = 0  # Magnitud máxima del gradiente para ver la convergencia

        for i in range(1, len(Q) - 1):
            # Calcula el gradiente para cada punto (excepto los extremos)
            gradient = alpha * (2 * P[i] - P[i - 1] - P[i + 1]) + beta * (P[i] - Q[i])

            # Actualiza el camino suavizado P
            P[i] -= epsilon * gradient

            # Calcula la magnitud máxima del cambio para verificar la convergencia
            max_delta = max(max_delta, np.linalg.norm(gradient))

        # Verifica si el gradiente es suficientemente pequeño para detenerse
        if max_delta < tol:
            print(f"Converged after {steps} steps.")
            break

        steps += 1
        # Mostrar progreso en la terminal para depuración
        print(f"Step {steps} - Max Gradient Delta: {max_delta}")

    return P

def callback_smooth_path(req):
    global msg_smooth_path
    # Manejar posibles errores si los parámetros no son proporcionados
    alpha = rospy.get_param('~alpha', 0.9)  # Parámetro alpha por defecto 0.9
    beta  = rospy.get_param('~beta', 0.1)   # Parámetro beta por defecto 0.1
    steps = rospy.get_param('~steps', 10000)  # Máximo de pasos, por defecto 10000
    print(f"Smoothing path with params: [alpha: {alpha}, beta: {beta}, steps: {steps}]")

    # Tiempo de inicio para medir el tiempo del suavizado
    start_time = rospy.Time.now()

    # Suavizar el camino
    Q = np.asarray([[p.pose.position.x, p.pose.position.y] for p in req.path.poses])
    P = smooth_path(Q, alpha, beta, steps)

    # Tiempo de fin para medir el tiempo del suavizado
    end_time = rospy.Time.now()
    print(f"Path smoothed after {1000 * (end_time - start_time).to_sec()} ms")

    # Asignar el camino suavizado al mensaje para publicarlo
    msg_smooth_path.poses = []
    for i in range(len(req.path.poses)):
        msg_smooth_path.poses.append(PoseStamped(pose=Pose(position=Point(x=P[i, 0], y=P[i, 1]))))

    return ProcessPathResponse(processed_path=msg_smooth_path)

def main():
    global msg_smooth_path
    print(f"PATH SMOOTHING - {NAME}")
    
    # Inicializar el nodo de ROS
    rospy.init_node("path_smoothing", anonymous=True)

    # Crear el servicio de suavizado de caminos
    rospy.Service('/path_planning/smooth_path', ProcessPath, callback_smooth_path)
    
    # Publicador para el camino suavizado
    pub_path = rospy.Publisher('/path_planning/smooth_path', Path, queue_size=10)
    
    # Configuración de la tasa de publicación
    loop = rospy.Rate(1)

    # Inicializar el mensaje de camino suavizado
    msg_smooth_path = Path()
    msg_smooth_path.header.frame_id = "map"

    # Publicar el camino suavizado en un bucle
    while not rospy.is_shutdown():
        pub_path.publish(msg_smooth_path)
        loop.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass