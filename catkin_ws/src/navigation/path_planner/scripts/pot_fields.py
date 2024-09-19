#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-1
# OBSTACLE AVOIDANCE BY POTENTIAL FIELDS
#
# Instructions:
# Complete the code to implement obstacle avoidance by potential fields
# using the attractive and repulsive fields technique.
# Tune the constants alpha and beta to get a smooth movement. 
#

import rospy
import tf
import math
import numpy
from geometry_msgs.msg import Twist, PoseStamped, Point, Vector3
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan

listener    = None
pub_cmd_vel = None
pub_markers = None
laser_readings = None
v_max = 0.6
w_max = 1.0

NAME = "Alexis Pizano Bavo"

def calculate_control(goal_x, goal_y, alpha, beta):
    error_a = math.atan2(goal_y, goal_x)  # Error angular basado en la posición del objetivo
    
    v = v_max * math.exp(-error_a * error_a / alpha)  # Control de velocidad lineal
    w = w_max * (2 / (1 + math.exp(-error_a / beta)) - 1)  # Control de velocidad angular
    
    return [v, w]

def attraction_force(goal_x, goal_y, eta):
    # La fuerza de atracción es proporcional a la distancia al objetivo
    distance = math.sqrt(goal_x**2 + goal_y**2)
    force_x = -eta * goal_x / distance
    force_y = -eta * goal_y / distance
    
    return numpy.asarray([force_x, force_y])

def rejection_force(laser_readings, zeta, d0):
    N = len(laser_readings)
    if N == 0:
        return numpy.asarray([0, 0])

    force_x, force_y = 0, 0
    for distance, angle in laser_readings:
        if distance < d0:
            # Fuerza de repulsión proporcional a la inversa de la distancia al obstáculo
            rho = (1/distance - 1/d0)
            force_x += zeta * rho * math.cos(angle)
            force_y += zeta * rho * math.sin(angle)
    
    force_x /= N
    force_y /= N
    
    return numpy.asarray([force_x, force_y])

def move_by_pot_fields(global_goal_x, global_goal_y, epsilon, tol, eta, zeta, d0, alpha, beta):
    goal_x, goal_y = get_goal_point_wrt_robot(global_goal_x, global_goal_y)
    
    while math.sqrt(goal_x**2 + goal_y**2) > tol and not rospy.is_shutdown():
        # Calcula las fuerzas de atracción y repulsión
        Fa = attraction_force(goal_x, goal_y, eta)
        Fr = rejection_force(laser_readings, zeta, d0)
        
        # Suma las fuerzas resultantes
        F = Fa + Fr
        
        # Gradiente descendente: Calcula la nueva posición
        P = -epsilon * F
        
        # Calcula las velocidades de control
        v, w = calculate_control(goal_x, goal_y, alpha, beta)
        
        # Publica las velocidades y fuerzas
        publish_speed_and_forces(v, w, Fa, Fr, F)
        
        # Actualiza la posición del objetivo con respecto al robot
        goal_x, goal_y = get_goal_point_wrt_robot(global_goal_x, global_goal_y)
    
    return
        

def get_goal_point_wrt_robot(goal_x, goal_y):
    robot_x, robot_y, robot_a = get_robot_pose(listener)
    delta_x = goal_x - robot_x
    delta_y = goal_y - robot_y
    goal_x =  delta_x*math.cos(robot_a) + delta_y*math.sin(robot_a)
    goal_y = -delta_x*math.sin(robot_a) + delta_y*math.cos(robot_a)
    return [goal_x, goal_y]

def get_robot_pose(listener):
    try:
        ([x, y, z], [qx,qy,qz,qw]) = listener.lookupTransform('map', 'base_link', rospy.Time(0))
        return [x, y, 2*math.atan2(qz, qw)]
    except:
        return [0,0,0]

def publish_speed_and_forces(v, w, Fa, Fr, F):
    loop = rospy.Rate(20)
    pub_cmd_vel.publish(Twist(linear=Vector3(x=v), angular=Vector3(z=w)))
    pub_markers.publish(get_force_marker(Fa[0], Fa[1], [0.0, 0.0, 1.0, 1.0], 0))
    pub_markers.publish(get_force_marker(Fr[0], Fr[1], [1.0, 0.0, 0.0, 1.0], 1))
    pub_markers.publish(get_force_marker(F [0], F [1], [0.0, 0.6, 0.0, 1.0], 2))
    loop.sleep()

def get_force_marker(force_x, force_y, color, id):
    mrk = Marker()
    mrk.header.frame_id = "base_link"
    mrk.header.stamp = rospy.Time.now()
    mrk.ns = "pot_fields"
    mrk.id = id
    mrk.type = Marker.ARROW
    mrk.action = Marker.ADD
    mrk.pose.orientation.w = 1
    mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = color
    mrk.scale.x, mrk.scale.y, mrk.scale.z = [0.07, 0.1, 0.15]
    mrk.points.append(Point(x=0, y=0))
    mrk.points.append(Point(x=-force_x, y=-force_y))
    return mrk

def callback_scan(msg):
    global laser_readings
    laser_readings = [[msg.ranges[i], msg.angle_min+i*msg.angle_increment] for i in range(len(msg.ranges))]

def callback_pot_fields_goal(msg):
    [goal_x, goal_y] = [msg.pose.position.x, msg.pose.position.y]
    print("Moving to goal point " + str([goal_x, goal_y]) + " by potential fields")
    epsilon = rospy.get_param('~epsilon', 0.5)
    tol     = rospy.get_param('~tol', 0.5)
    eta     = rospy.get_param('~eta', 2.0)
    zeta    = rospy.get_param('~zeta', 6.0)
    d0      = rospy.get_param('~d0', 1.0)
    alpha   = rospy.get_param('~alpha', 0.5)
    beta    = rospy.get_param('~beta', 0.5)
    move_by_pot_fields(goal_x, goal_y, epsilon, tol, eta, zeta, d0, alpha, beta)
    pub_cmd_vel.publish(Twist())
    print("Global goal point reached")

def main():
    global listener, pub_cmd_vel, pub_markers
    print("POTENTIAL FIELDS - " + NAME)
    rospy.init_node("pot_fields")
    rospy.Subscriber("/hardware/scan", LaserScan, callback_scan)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_pot_fields_goal)
    pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist,  queue_size=10)
    pub_markers = rospy.Publisher('/navigation/pot_field_markers', Marker, queue_size=10)
    listener = tf.TransformListener()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

    
