#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-1
# PATH FOLLOWING
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
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanRequest
from navig_msgs.srv import ProcessPath, ProcessPathRequest
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point

NAME = "PACHECO SALGADO MAURICIO"

pub_goal_reached = None
pub_cmd_vel = None
loop = None
listener = None

def calculate_control(robot_x, robot_y, robot_a, goal_x, goal_y, alpha, beta, v_max, w_max):
    v, w = 0, 0
    #
    # TODO:
    # Implement the control law given by:
    #
    error_a = math.atan2(goal_y - robot_y, goal_x - robot_x) - robot_a
    error_a = (error_a + math.pi) % (2 * math.pi) - math.pi
    v = v_max * math.exp(-error_a * error_a / alpha)
    w = w_max * (2 / (1 + math.exp(-error_a / beta)) - 1)
    #
    # where error_a is the angle error
    # and v_max, w_max, alpha and beta, are tunning constants.
    # Remember to keep error angle in the interval (-pi,pi]
    # Return the tuple [v,w]
    #

    return [v, w]

def follow_path(path, alpha, beta, v_max, w_max):
    # Listas para almacenar las coordenadas y velocidades
    ruta_deseada = []
    ruta_real = []
    velocidades_lineales = []
    velocidades_angulares = []

    idx = 0
    Pg = path[idx]
    Pr, robot_a = get_robot_pose()

    while numpy.linalg.norm(path[-1] - Pr) > 0.1 and not rospy.is_shutdown():
        # Almacenar la ruta deseada (punto objetivo actual)
        ruta_deseada.append(Pg)

        v, w = calculate_control(Pr[0], Pr[1], robot_a, Pg[0], Pg[1], alpha, beta, v_max, w_max)

        # Almacenar las velocidades
        velocidades_lineales.append(v)
        velocidades_angulares.append(w)

        publish_twist(v, w)
        Pr, robot_a = get_robot_pose()

        # Almacenar la ruta real (posición actual del robot)
        ruta_real.append(Pr)

        if numpy.linalg.norm(Pg - Pr) < 0.3:
            idx = min(idx + 1, len(path) - 1)
            Pg = path[idx]

    # Guardar la ruta deseada en un archivo CSV
    with open('ruta_deseada.csv', 'w') as archivo_csv:
        archivo_csv.write("X,Y\n")  # Encabezados
        for punto in ruta_deseada:
            archivo_csv.write(f"{punto[0]},{punto[1]}\n")

    # Guardar la ruta real en un archivo CSV
    with open('ruta_real.csv', 'w') as archivo_csv:
        archivo_csv.write("X,Y\n")  # Encabezados
        for punto in ruta_real:
            archivo_csv.write(f"{punto[0]},{punto[1]}\n")

    # Guardar las velocidades en un archivo CSV
    with open('velocidades.csv', 'w') as archivo_csv:
        archivo_csv.write("Lineal,Angular\n")  # Encabezados
        for v, w in zip(velocidades_lineales, velocidades_angulares):
            archivo_csv.write(f"{v},{w}\n")

    return ruta_deseada, ruta_real, velocidades_lineales, velocidades_angulares

def publish_twist(v, w):
    loop = rospy.Rate(20)
    msg = Twist()
    msg.linear.x = v
    msg.angular.z = w
    pub_cmd_vel.publish(msg)
    loop.sleep()

def callback_global_goal(msg):
    print("Calculating path from robot pose to " + str([msg.pose.position.x, msg.pose.position.y]))
    [robot_x, robot_y], robot_a = get_robot_pose()
    req = GetPlanRequest(goal=PoseStamped(pose=msg.pose))
    req.start.pose.position = Point(x=robot_x, y=robot_y)
    path = rospy.ServiceProxy('/path_planning/plan_path', GetPlan)(req).plan
    if len(path.poses) < 2:
        print("Cannot calculate path")
        return
    try:
        smooth_path = rospy.ServiceProxy('/path_planning/smooth_path', ProcessPath)(ProcessPathRequest(path=path)).processed_path
        path = smooth_path
    except:
        pass
    v_max = rospy.get_param("~v_max", 0.8)
    w_max = rospy.get_param("~w_max", 1.0)
    alpha = rospy.get_param("~alpha", 1.0)
    beta = rospy.get_param("~beta", 0.1)
    print("Following path with [v_max, w_max, alpha, beta]=" + str([v_max, w_max, alpha, beta]))
    follow_path([numpy.asarray([p.pose.position.x, p.pose.position.y]) for p in path.poses], alpha, beta, v_max, w_max)
    pub_cmd_vel.publish(Twist())
    pub_goal_reached.publish(True)
    print("Global goal point reached")

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
    print("Waiting for service for path planning...")
    rospy.wait_for_service('/path_planning/plan_path')
    print("Service for path planning is now available.")
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
