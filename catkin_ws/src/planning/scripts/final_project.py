#!/usr/bin/env python3
#
# MOBILE ROBOTS - FI-UNAM, 2025-1
# FINAL PROJECT - SIMPLE SERVICE ROBOT
# 
# Instructions:
# Write the code necessary to make the robot to perform the following possible commands:
# * Robot take the <pringles|drink> to the <table|kitchen>
# You can choose where the table and kitchen are located within the map.
# The Robot must recognize the orders using speech recognition.
# Entering the command by text or similar way is not allowed.
# The Robot must announce the progress of the action using speech synthesis,
# for example: I'm going to grab..., I'm going to navigate to ..., I arrived to..., etc.
#

import rospy
import tf
import math
import time
from std_msgs.msg import String, Float64MultiArray, Float64, Bool
from nav_msgs.msg import Path
from nav_msgs.srv import GetPlan, GetPlanRequest
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point, PointStamped
from trajectory_msgs.msg import JointTrajectory
from sound_play.msg import SoundRequest
from vision_msgs.srv import *
from manip_msgs.srv import *
from hri_msgs.msg import *

NAME = "Pizano Bravo Alexis"

# Global variables
recognized_speech = ""
new_task = False
executing_task = False
goal_reached = False
current_state = "SM_INIT"
task_object = ""
task_location = []

# Callback for recognized speech
def callback_recognized_speech(msg):
    global recognized_speech, new_task, executing_task, current_state, task_object, task_location
    if executing_task:
        return
    recognized_speech = msg.hypothesis[0].upper()  # Convert to uppercase for uniform comparison
    print("New command recognized: " + recognized_speech)
    task_object, task_location = parse_command(recognized_speech)  # Extract object and location
    new_task = True
    current_state = "SM_PLAN"

# Callback for goal reached
def callback_goal_reached(msg):
    global goal_reached
    goal_reached = msg.data
    print("Received goal reached: " + str(goal_reached))

# Parse the command to extract the object and location
def parse_command(cmd):
    obj = "pringles" if "PRINGLES" in cmd else "drink"
    loc = [8.0, 8.5] if "TABLE" in cmd else [3.22, 9.72]
    return obj, loc

# Function to move the left arm
def move_left_arm_with_trajectory(Q):
    global pubLaGoalTraj
    pubLaGoalTraj.publish(Q)
    time.sleep(0.05 * len(Q.points) + 2)

# Function to move the left gripper
def move_left_gripper(q):
    global pubLaGoalGrip
    pubLaGoalGrip.publish(q)
    time.sleep(1.0)

# Function to send a global goal position
def go_to_goal_pose(goal_x, goal_y):
    global pubGoalPose
    goal_pose = PoseStamped()
    goal_pose.pose.orientation.w = 1.0
    goal_pose.pose.position.x = goal_x
    goal_pose.pose.position.y = goal_y
    pubGoalPose.publish(goal_pose)

# Function to synthesize speech
def say(text):
    global pubSay
    msg = SoundRequest()
    msg.sound = -3
    msg.command = 1
    msg.volume = 1.0
    msg.arg2 = "voice_kal_diphone"
    msg.arg = text
    pubSay.publish(msg)

# Find an object using the vision service
def find_object(object_name):
    clt_find_object = rospy.ServiceProxy("/vision/obj_reco/detect_and_recognize_object", RecognizeObject)
    req_find_object = RecognizeObjectRequest()
    req_find_object.point_cloud = rospy.wait_for_message("/camera/depth_registered/points", PointCloud2)
    req_find_object.name = object_name
    resp = clt_find_object(req_find_object)
    return [resp.recog_object.pose.position.x, resp.recog_object.pose.position.y, resp.recog_object.pose.position.z]

# Transform a point from one frame to another
def transform_point(x, y, z, source_frame="realsense_link", target_frame="shoulders_left_link"):
    listener = tf.TransformListener()
    listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(4.0))
    obj_p = PointStamped()
    obj_p.header.frame_id = source_frame
    obj_p.header.stamp = rospy.Time(0)
    obj_p.point.x, obj_p.point.y, obj_p.point.z = x, y, z
    obj_p = listener.transformPoint(target_frame, obj_p)
    return [obj_p.point.x, obj_p.point.y, obj_p.point.z]

# Calculate inverse kinematics for the left arm
def calculate_inverse_kinematics_left(x, y, z, roll, pitch, yaw):
    req_ik = InverseKinematicsPose2TrajRequest()
    req_ik.x = x
    req_ik.y = y
    req_ik.z = z
    req_ik.roll = roll
    req_ik.pitch = pitch
    req_ik.yaw = yaw
    req_ik.duration = 0
    req_ik.time_step = 0.05
    req_ik.initial_guess = []
    clt = rospy.ServiceProxy("/manipulation/la_ik_trajectory", InverseKinematicsPose2Traj)
    resp = clt(req_ik)
    return resp.articular_trajectory

def main():
    global new_task, recognized_speech, executing_task, goal_reached
    global pubLaGoalPose, pubLaGoalTraj, pubGoalPose, pubCmdVel, pubSay
    print("FINAL PROJECT - " + NAME)
    rospy.init_node("final_project")
    rospy.Subscriber('/hri/sp_rec/recognized', RecognizedSpeech, callback_recognized_speech)
    rospy.Subscriber('/navigation/goal_reached', Bool, callback_goal_reached)
    pubGoalPose = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    pubCmdVel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pubSay = rospy.Publisher('/hri/speech_generator', SoundRequest, queue_size=10)
    pubLaGoalPose = rospy.Publisher("/hardware/left_arm/goal_pose", Float64MultiArray, queue_size=10)
    pubLaGoalTraj = rospy.Publisher("/manipulation/la_q_trajectory", JointTrajectory, queue_size=10)
    listener = tf.TransformListener()
    loop = rospy.Rate(10)
    print("Waiting for services...")
    rospy.wait_for_service('/manipulation/la_ik_trajectory')
    rospy.wait_for_service('/vision/obj_reco/detect_and_recognize_object')
    print("Services are now available.")

    executing_task = False
    current_state = "SM_INIT"
    new_task = False
    goal_reached = False
    while not rospy.is_shutdown():
        if current_state == "SM_INIT":
            print("State: SM_INIT")
            if new_task:
                new_task = False
                current_state = "SM_PLAN"
            loop.sleep()

        elif current_state == "SM_PLAN":
            print(f"State: SM_PLAN - Object: {task_object}, Location: {task_location}")
            say(f"I'm going to grab the {task_object} and take it to the destination.")
            current_state = "SM_GRAB"

        elif current_state == "SM_GRAB":
            print("State: SM_GRAB")
            obj_coords = find_object(task_object)
            obj_transformed = transform_point(*obj_coords)
            traj = calculate_inverse_kinematics_left(*obj_transformed, 0, 0, 0)
            move_left_arm_with_trajectory(traj)
            move_left_gripper(0.5)
            say(f"I grabbed the {task_object}.")
            current_state = "SM_NAVIGATE"

        elif current_state == "SM_NAVIGATE":
            print("State: SM_NAVIGATE")
            go_to_goal_pose(task_location[0], task_location[1])
            while not goal_reached:
                loop.sleep()
            say("I arrived at the destination.")
            current_state = "SM_PLACE"

        elif current_state == "SM_PLACE":
            print("State: SM_PLACE")
            move_left_gripper(1.0)
            say(f"I placed the {task_object} at the destination.")
            current_state = "SM_DONE"

        elif current_state == "SM_DONE":
            print("State: SM_DONE")
            executing_task = False
            current_state = "SM_INIT"

        loop.sleep()

if __name__ == '__main__':
    main()

