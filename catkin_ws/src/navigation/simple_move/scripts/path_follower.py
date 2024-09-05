#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-1
# PATH FOLLOWING WITH PLOTTING
#
# Instructions:
# Write the code necessary to move the robot along a given path.
# Consider a differential base. Max linear and angular speeds
# must be 0.8 and 1.0 respectively.
#

import rospy
import tf
import math
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import csv
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanRequest
from navig_msgs.srv import ProcessPath, ProcessPathRequest
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point

NAME = "FULL NAME"

pub_goal_reached = None
pub_cmd_vel = None
loop = None
listener = None

# Variables para almacenar datos
positions = []
velocities = []

def calculate_control(robot_x, robot_y, robot_a, goal_x, goal_y, alpha, beta, v_max, w_max):
    error_a = math.atan2(goal_y - robot_y, goal_x - robot_x) - robot_a
    error_a = (error_a + math.pi) % (2 * math.pi) - math.pi  # Normalizar a [-pi, pi]
    
    # Reducir velocidades al acercarse al objetivo
    dist_to_goal = math.sqrt((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2)
    
    if dist_to_goal < 0.5:  # Ajusta la distancia para reducir la velocidad
        v_max *= dist_to_goal
        w_max *= dist_to_goal
    
    v = v_max * math.exp(-error_a * error_a / alpha)
    w = w_max * (2 / (1 + math.exp(-error_a / beta)) - 1)
    
    return [v, w]

def follow_path(path, alpha, beta, v_max, w_max):
    idx = 0
    Pg = path[idx]
    Pr, robot_a = get_robot_pose()
    
    # Loop para seguir el camino
    while numpy.linalg.norm(path[-1] - Pr) > 0.1 and not rospy.is_shutdown():
        v, w = calculate_control(Pr[0], Pr[1], robot_a, Pg[0], Pg[1], alpha, beta, v_max, w_max)
        publish_twist(v, w)
        Pr, robot_a = get_robot_pose()

        # Guardar posición y velocidades
        positions.append((Pr[0], Pr[1]))
        velocities.append((v, w))

        if numpy.linalg.norm(Pg - Pr) < 0.3:  # Ajustar la tolerancia para cambiar al siguiente punto
            idx = min(idx + 1, len(path) - 1)
            Pg = path[idx]

    # Detener el robot al llegar al objetivo
    pub_cmd_vel.publish(Twist())
    
    # Guardar datos en CSV
    save_to_csv()

    # Graficar resultados
    plot_results(path)

    return

def publish_twist(v, w):
    loop = rospy.Rate(20)  # Ajustar la frecuencia de publicación si es necesario
    msg = Twist()
    msg.linear.x = v
    msg.angular.z = w
    pub_cmd_vel.publish(msg)
    loop.sleep()

def plot_results(path):
    # Convertir los datos a DataFrame para un manejo más fácil
    df_positions = pd.DataFrame(positions, columns=['x', 'y'])
    df_velocities = pd.DataFrame(velocities, columns=['linear', 'angular'])
    
    # Graficar la trayectoria
    plt.figure()
    plt.plot(df_positions['x'], df_positions['y'], label="Robot Path")
    plt.plot([p[0] for p in path], [p[1] for p in path], label="Planned Path", linestyle="--")
    plt.title("Robot Path vs Planned Path")
    plt.legend()
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()

    # Graficar las velocidades
    plt.figure()
    plt.plot(df_velocities['linear'], label="Linear Velocity")
    plt.plot(df_velocities['angular'], label="Angular Velocity")
    plt.title("Velocities over Time")
    plt.legend()
    plt.xlabel("Time (steps)")
    plt.ylabel("Velocity")
    plt.grid()

    # Mostrar gráficos
    plt.show()

def save_to_csv():
    # Guardar las posiciones y velocidades en un archivo CSV
    with open('robot_data.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['X Position', 'Y Position', 'Linear Velocity', 'Angular Velocity'])
        for pos, vel in zip(positions, velocities):
            writer.writerow([pos[0], pos[1], vel[0], vel[1]])
    print("Datos guardados en robot_data.csv")

def callback_global_goal(msg):
    print("Calculando el camino desde la posición del robot hasta " + str([msg.pose.position.x, msg.pose.position.y]))
    [robot_x, robot_y], robot_a = get_robot_pose()
    req = GetPlanRequest(goal=PoseStamped(pose=msg.pose))
    req.start.pose.position = Point(x=robot_x, y=robot_y)
    path = rospy.ServiceProxy('/path_planning/plan_path', GetPlan)(req).plan
    if len(path.poses) < 2:
        print("No se puede calcular la ruta")
        return
    try:
        smooth_path = rospy.ServiceProxy('/path_planning/smooth_path', ProcessPath)(ProcessPathRequest(path=path)).processed_path
        path = smooth_path
    except:
        pass

    # Tomar los parámetros desde la terminal
    v_max = rospy.get_param("~v_max", 0.8)
    w_max = rospy.get_param("~w_max", 1.0)
    alpha = rospy.get_param("~alpha", 1.0)
    beta = rospy.get_param("~beta", 0.5)
    
    print(f"Following path with [v_max, w_max, alpha, beta]={[v_max, w_max, alpha, beta]}")
    follow_path([numpy.asarray([p.pose.position.x, p.pose.position.y]) for p in path.poses], alpha, beta, v_max, w_max)
    pub_goal_reached.publish(True)
    print("Objetivo global alcanzado")

def get_robot_pose():
    try:
        ([x, y, z], [qx, qy, qz, qw]) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        return numpy.asarray([x, y]), 2 * math.atan2(qz, qw)
    except:
        return numpy.asarray([0, 0]), 0

def main():
    global pub_cmd_vel, pub_goal_reached, loop, listener
    print("PATH FOLLOWING - " + NAME)
    rospy.init_node("path_follower")
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_global_goal)
    pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pub_goal_reached = rospy.Publisher('/navigation/goal_reached', Bool, queue_size=10)
    listener = tf.TransformListener()
    loop = rospy.Rate(10)
    print("Esperando servicio de planificación de ruta...")
    rospy.wait_for_service('/path_planning/plan_path')
    print("Servicio de planificación de rutas disponible.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
