#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-1
# INVERSE KINEMATICS USING NEWTON-RAPHSON
#
# Instructions:
# Calculate the inverse kinematics using
# the Newton-Raphson method for root finding.
# Modify only sections marked with the 'TODO' comment
#

import math
import sys
import rospy
import numpy
import tf
import tf.transformations as tft
import urdf_parser_py.urdf
from std_msgs.msg import Float64MultiArray
from manip_msgs.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

prompt = ""
NAME = "Torres Anguiano Azael Arturo"

def forward_kinematics(q, T, W):
    x, y, z, R, P, Y = 0, 0, 0, 0, 0, 0
    #
    # TODO:
    # Calculate the forward kinematics given the set of seven angles 'q'
    # You can use the following steps:
    #     H = I   # Assing to H a 4x4 identity matrix
    #     for all qi in q:
    #         H = H * Ti * Ri
    #     H = H * Ti[7]
    #     Get xyzRPY from the resulting Homogeneous Transformation 'H'
    # Where:
    #     Ti are the Homogeneous Transforms from frame i to frame i-1 when joint i is at zero position
    #     Ri are the Homogeneous Transforms with zero-translation and rotation qi around axis Wi.
    #     Ti[7] is the final Homogeneous Transformation from gripper center to joint 7.
    # Hints:
    #     Use the tft.identity_matrix() function to get the 4x4 I
    #     Use the tft.concatenate_matrices() function for multiplying Homogeneous Transformations
    #     Use the tft.rotation_matrix() matrices Ri.
    #     Use the tft.euler_from_matrix() function to get RPY from matrix H
    #     Check online documentation of these functions:
    #     http://docs.ros.org/en/jade/api/tf/html/python/transformations.html
    #
    H = tft.identity_matrix() 
    for i in range(len(q)):
        Ti = T[i]  
        Ri = tft.rotation_matrix(q[i], W[i])  
        H = tft.concatenate_matrices(H, Ti, Ri)
    H = tft.concatenate_matrices(H, T[-1])  

    # Extraer traslación y rotación de la matriz H
    x, y, z = H[:3, 3]
    R, P, Y = tft.euler_from_matrix(H)  # Obtener RPY desde H
    return numpy.asarray([x, y, z, R, P, Y])

def jacobian(q, T, W):
    delta_q = 0.000001
    #
    # TODO:
    # Calculate the Jacobian given a kinematic description Ti and Wi
    # where:
    # Ti are the Homogeneous Transformations from frame i to frame i-1 when joint i is at zero position
    # Wi are the axis of rotation of i-th joint
    # Use the numeric approximation:   f'(x) = (f(x+delta) - f(x-delta))/(2*delta)
    #
    # You can do the following steps:
    #     J = matrix of 6x7 full of zeros
    #     q_next = [q1+delta       q2        q3   ....     q7
    #                  q1       q2+delta     q3   ....     q7
    #                              ....
    #                  q1          q2        q3   ....   q7+delta]
    #     q_prev = [q1-delta       q2        q3   ....     q7
    #                  q1       q2-delta     q3   ....     q7
    #                              ....
    #                  q1          q2        q3   ....   q7-delta]
    #     FOR i = 1,..,7:
    #           i-th column of J = ( FK(i-th row of q_next) - FK(i-th row of q_prev) ) / (2*delta_q)
    #     RETURN J
    #
    J = numpy.zeros((6, 7))  
    for i in range(len(q)):
        q_next = q.copy()
        q_prev = q.copy()
        q_next[i] += delta_q
        q_prev[i] -= delta_q

        FK_next = forward_kinematics(q_next, T, W)
        FK_prev = forward_kinematics(q_prev, T, W)

        J[:, i] = (FK_next - FK_prev) / (2 * delta_q)
    return J

def inverse_kinematics(x, y, z, roll, pitch, yaw, T, W, init_guess=numpy.zeros(7), max_iter=20):
    pd = numpy.asarray([x, y, z, roll, pitch, yaw])
    #
    # TODO:
    # Solve the IK problem given a kinematic description (Ti, Wi) and a desired configuration.
    # where:
    # Ti are the Homogeneous Transformations from frame i to frame i-1 when joint i is at zero position
    # Wi are the axis of rotation of i-th joint
    # Use the Newton-Raphson method for root finding. (Find the roots of equation FK(q) - pd = 0)
    # You can do the following steps:
    #
    #    Set an initial guess for joints 'q'
    #    Calculate Forward Kinematics 'p' by calling the corresponding function
    #    Calcualte error = p - pd
    #    Ensure orientation angles of error are in [-pi,pi]
    #    WHILE |error| > TOL and iterations < maximum iterations:
    #        Calculate Jacobian
    #        Update q estimation with q = q - pseudo_inverse(J)*error
    #        Ensure all angles q are in [-pi,pi]
    #        Recalculate forward kinematics p
    #        Recalculate error and ensure angles are in [-pi,pi]
    #        Increment iterations
    #    Set success if maximum iterations were not exceeded and calculated angles are in valid range
    #    Return calculated success and calculated q
    #
    q = init_guess
    iterations = 0
    TOL = 1e-6  
    while iterations < max_iter:
        p = forward_kinematics(q, T, W)
        error = p - pd
        error[3:] = numpy.mod(error[3:] + numpy.pi, 2 * numpy.pi) - numpy.pi  

        if numpy.linalg.norm(error) < TOL:
            return True, q

        J = jacobian(q, T, W)

        dq = numpy.dot(numpy.linalg.pinv(J), error)
        q -= dq

        q = numpy.mod(q + numpy.pi, 2 * numpy.pi) - numpy.pi
        iterations += 1

    return False, q

def get_polynomial_trajectory_multi_dof(Q_start, Q_end, duration=1.0, time_step=0.05):
    clt = rospy.ServiceProxy("/manipulation/polynomial_trajectory", GetPolynomialTrajectory)
    req = GetPolynomialTrajectoryRequest()
    req.p1 = Q_start
    req.p2 = Q_end
    req.duration = duration
    req.time_step = 0.05
    resp = clt(req)
    Q = []
    T = []
    for p in resp.trajectory.points:
        Q.append(p.positions)
        T.append(p.time_from_start.to_sec())
    return numpy.asarray(Q), numpy.asarray(T)

def get_model_info(joint_names):
    robot_model = urdf_parser_py.urdf.URDF.from_parameter_server()
    joints = []
    transforms = []
    for name in joint_names:
        for joint in robot_model.joints:
            if joint.name == name:
                joints.append(joint)
    for joint in joints:
        T = tft.translation_matrix(joint.origin.xyz)
        R = tft.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2])
        transforms.append(tft.concatenate_matrices(T,R))
    return joints, transforms

def angles_in_joint_limits(q):
    for i in range(len(q)):
        if q[i] < joints[i].limit.lower or q[i] > joints[i].limit.upper:
            print(prompt+"Articular position out of joint bounds")
            return False
    return True

def callback_forward_kinematics(req):
    if len(req.q) != 7:
        print(prompt+"By the moment, only 7-DOF arm is supported")
        return False
    resp = ForwardKinematicsResponse()
    W = [joints[i].axis for i in range(len(joints))]  
    resp.x,resp.y,resp.z,resp.roll,resp.pitch,resp.yaw = forward_kinematics(req.q, transforms, W)
    return resp

def get_trajectory_time(p1, p2, speed_factor):
    p1 = numpy.asarray(p1)
    p2 = numpy.asarray(p2)
    m = max(numpy.absolute(p1 - p2))
    return m/speed_factor + 0.5

def callback_ik_for_trajectory(req):
    global max_iterations
    Pd = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]
    print(prompt+"Calculating IK and trajectory for " + str(Pd))
    if len(req.initial_guess) <= 0 or req.initial_guess == None:
        initial_guess = rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray)
        initial_guess = initial_guess.data
    else:
        initial_guess = req.initial_guess
    W = [joints[i].axis for i in range(len(joints))]  
    p1 = forward_kinematics(initial_guess, transforms, W)
    p2 = [req.x, req.y, req.z, req.roll, req.pitch, req.yaw]
    t  = req.duration if req.duration > 0 else get_trajectory_time(p1, p2, 0.25)
    dt = req.time_step if req.time_step > 0 else 0.05
    X,T = get_polynomial_trajectory_multi_dof(p1, p2, duration=t, time_step=dt)
    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()
    q = initial_guess
    for i in range(len(X)):
        x, y, z, roll, pitch, yaw = X[i]
        success, q = inverse_kinematics(x, y, z, roll, pitch, yaw, transforms, W, q, max_iterations)
        if not success:
            return False
        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start = rospy.Duration.from_sec(T[i])
        trj.points.append(p)
    resp = InverseKinematicsPose2TrajResponse()
    resp.articular_trajectory = trj
    return resp
    
def callback_ik_for_pose(req):
    global max_iterations
    [x,y,z,R,P,Y] = [req.x,req.y,req.z,req.roll,req.pitch,req.yaw]
    print(prompt+"Calculating inverse kinematics for pose: " + str([x,y,z,R,P,Y]))
    if len(req.initial_guess) <= 0 or req.initial_guess == None:
        init_guess = rospy.wait_for_message("/hardware/arm/current_pose", Float64MultiArray, 5.0)
        init_guess = initial_guess.data
    else:
        init_guess = req.initial_guess
    resp = InverseKinematicsPose2PoseResponse()
    success, q = inverse_kinematics(x, y, z, R, P, Y, init_guess, max_iterations)
    if not success:
        return False
    resp.q = q
    return resp        

def main():
    global joint_names, max_iterations, joints, transforms, prompt
    print("INITIALIZING INVERSE KINEMATIC NODE - " + NAME)
    rospy.init_node("ik_geometric")
    prompt = rospy.get_name().upper() + ".->"
    joint_names    = rospy.get_param("~joint_names", [])
    max_iterations = rospy.get_param("~max_iterations", 20)
    print(prompt+"Using joints: " + str(joint_names))
    joints, transforms = get_model_info(joint_names)
    rospy.Service("/manipulation/ik_geometric/ik_for_trajectory", InverseKinematicsPose2Traj, callback_ik_for_trajectory)
    rospy.Service("/manipulation/ik_geometric/ik_for_pose", InverseKinematicsPose2Pose, callback_ik_for_pose)
    rospy.Service("/manipulation/ik_geometric/fk", ForwardKinematics, callback_forward_kinematics)
    rospy.spin()

if __name__ == "__main__":
    main()
