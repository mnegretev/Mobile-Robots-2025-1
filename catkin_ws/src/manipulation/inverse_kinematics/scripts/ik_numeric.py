#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2024-2
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
NAME = "Salazar Barrera Diego"
   
def forward_kinematics(q, T, W):
    x,y,z,R,P,Y = 0,0,0,0,0,0
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
    for i, qi in enumerate(q):
        Ti = T[i] 
        Ri = tft.rotation_matrix(qi,W[i])
        H = tft.concatenate_matrices(H,Ti)
        H = tft.concatenate_matrices(H,Ri)
    H = tft.concatenate_matrices(H,T[7])
    xyz = H[:3,3]
    RPY = tft.euler_from_matrix(H, axes='sxyz')
    
    x, y, z = xyz
    R, P, Y = RPY
 
    return numpy.asarray([x,y,z,R,P,Y])

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
    J = numpy.asarray([[0.0 for a in q] for i in range(6)])
    
    for i in range(7):
        q_next = numpy.copy(q)
        q_prev = numpy.copy(q)
        
        q_next[i] = q[i] + delta_q
        q_prev[i] = q[i] - delta_q
        
        FK_next = forward_kinematics(q_next,T,W)
        FK_prev = forward_kinematics(q_prev,T,W)    
        
        J[:3, i] = (FK_next[:3] - FK_prev[:3]) / (2 * delta_q)  # Position difference
        J[3:, i] = (FK_next[3:] - FK_prev[3:]) / (2 * delta_q)  # Orientation difference

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
    # Inicializa el error como un array de 6 ceros (para posición y orientación)
    error = numpy.zeros(6)
    q = init_guess
    iterations = 0
    success = False
    
    # Mientras el error sea grande y no se haya alcanzado el número máximo de iteraciones
    while numpy.linalg.norm(pd - forward_kinematics(q, T, W)) > 1e-6 and iterations < max_iter:
        # Calcula la cinemática directa 'p' llamando a la función correspondiente
        p = forward_kinematics(q, T, W)
        
        if len(p) >= 6 and len(pd) >= 6:
            # Asegúrate de que 'error' esté bien indexado
            error[:3] = p[:3] - pd[:3]  # Diferencia de la posición
            error[3:] = (p[3:] - pd[3:] + numpy.pi) % (2 * numpy.pi) - numpy.pi  # Diferencia de la orientación
        else:
            print("Error: p o pd tiene una longitud incorrecta.")
        
        # Calcula el Jacobiano
        J = jacobian(q, T, W)
        
        # Actualiza la estimación de 'q' con q = q - pseudo_inverse(J)*error
        J_pseudo_inverse = numpy.linalg.pinv(J)
        q = q - J_pseudo_inverse @ error
        
        # Asegúrate de que todos los ángulos de 'q' estén en el rango [-pi, pi]
        q = (q + numpy.pi) % (2 * numpy.pi) - numpy.pi
        
        # Recalcula la cinemática directa 'p' para el nuevo 'q'
        p = forward_kinematics(q, T, W)
        
        # Recalcula el error y asegúrate de que los ángulos estén en el rango [-pi, pi]
        error[:3] = p[:3] - pd[:3]
        error[3:] = (error[3:] + numpy.pi) % (2 * numpy.pi) - numpy.pi
        
        iterations += 1
        
    
    # Verificar si se completó la cinemática inversa y si se están cumpliendo los límites de los ángulos
    if iterations < max_iter and angles_in_joint_limits(q):
        success = True
    
    if success:
        print(f"Cinemática inversa de sub meta completa en {iterations} iteraciones.")
    else:
        print("Cinemática no completa")
    
    return success, q, iterations


   
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
    t = req.duration if req.duration > 0 else get_trajectory_time(p1, p2, 0.25)
    dt = req.time_step if req.time_step > 0 else 0.05
    X, T = get_polynomial_trajectory_multi_dof(p1, p2, duration=t, time_step=dt)

    total_iterations = 0  # Variable para acumular las iteraciones

    trj = JointTrajectory()
    trj.header.stamp = rospy.Time.now()

    q = initial_guess
    for i in range(len(X)):
        x, y, z, roll, pitch, yaw = X[i]
        success, q, iterations = inverse_kinematics(x, y, z, roll, pitch, yaw, transforms, W, q, max_iterations)
        
        if not success:  # Si no se puede encontrar una solución, retorna False
            print(f"Kinematic failed int: {total_iterations}")
            return False
        
        total_iterations += iterations  # Acumula el número total de iteraciones
        
        p = JointTrajectoryPoint()
        p.positions = q
        p.time_from_start = rospy.Duration.from_sec(T[i])
        trj.points.append(p)

    # Imprime el total de iteraciones después de procesar todas las sub-metas
    print(f"Total iterations for IK across all sub-goals: {total_iterations}")

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
    print(prompt+"Joint names: " + str(joint_names))
    print(prompt+"max_iterations: " + str(max_iterations))

    joints, transforms = get_model_info(joint_names)
    if not (len(joints) > 6 and len(transforms) > 6):
        print("Inverse kinematics.->Cannot get model info from parameter server")
        sys.exit(-1)

    rospy.Service("/manipulation/forward_kinematics"   , ForwardKinematics, callback_forward_kinematics)    
    rospy.Service("/manipulation/ik_trajectory"        , InverseKinematicsPose2Traj, callback_ik_for_trajectory)
    rospy.Service("/manipulation/ik_pose"              , InverseKinematicsPose2Pose, callback_ik_for_pose)
    #loop = rospy.Rate(10)
    loop = rospy.Rate(40)
    while not rospy.is_shutdown():
        loop.sleep()

if __name__ == '__main__':
    main()


