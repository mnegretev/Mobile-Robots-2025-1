#!/usr/bin/env python3

import rospy
import random
import math
import csv
from nav_msgs.srv import GetPlan, GetPlanRequest
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped

# Global lists to store distances, elapsed times and success
dist = []
time = []
success = []

def callback(data):
    print(data.data)
    # Store the elapsed time in the global list
    time.append(str(data.data))
    
def call_path_planning_service(start_pose, goal_pose):
    rospy.wait_for_service('/path_planning/plan_path')
    try:
        # Create the service proxy
        plan_service = rospy.ServiceProxy('/path_planning/plan_path', GetPlan)
        
        # Create a request
        request = GetPlanRequest()
        request.start = start_pose
        request.goal = goal_pose
        
        # Call the service
        response = plan_service(request)
        
        # Display the result
        num = len(response.plan.poses)
        print(f"Path Plan Response for goal ({goal_pose.pose.position.x}, {goal_pose.pose.position.y}):")
        print(f"Number of poses in the plan: {num}")
        if num > 0:
            # Show the success
            success.append(1)
        else:
            success.append(0)
            print("No path could be planned.")
    
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

def generate_random_goals(num_goals, x_range, y_range):
    goals = []
    for _ in range(num_goals):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = random.uniform(x_range[0], x_range[1])
        goal_pose.pose.position.y = random.uniform(y_range[0], y_range[1])
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        goals.append(goal_pose)
    return goals

def calculate_distance(pose1, pose2):
    """Calculates the Euclidean distance between two poses and stores it in a global list."""
    global dist
    dx = pose1.pose.position.x - pose2.pose.position.x
    dy = pose1.pose.position.y - pose2.pose.position.y
    distance = math.sqrt(dx ** 2 + dy ** 2)
    dist.append(distance)  # Store the distance in the global list
    return distance

def save_to_csv(filename):
    """Saves the distances, times, and successes to a CSV file."""
    dist.sort()
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Distance', 'Time (ms)', 'Success'])
        for d, t, s in zip(dist, time, success):
            writer.writerow([d, t, s])

if __name__ == "__main__":
    rospy.init_node('path_planning_client')
    
    # Subscribe to the topic publishing time
    rospy.Subscriber("time_topic", Float32, callback)

    # Define the start pose
    start_pose = PoseStamped()
    start_pose.header.frame_id = "map"
    start_pose.pose.position.x = 0.0
    start_pose.pose.position.y = 0.0
    start_pose.pose.position.z = 0.0
    start_pose.pose.orientation.x = 0.0
    start_pose.pose.orientation.y = 0.0
    start_pose.pose.orientation.z = 0.0
    start_pose.pose.orientation.w = 1.0

    # Generate 100 random goal poses
    num_goals = 100
    x_range = (0.0, 10.0)  # Range for the x coordinate
    y_range = (0.0, 10.0)  # Range for the y coordinate
    goal_poses = generate_random_goals(num_goals, x_range, y_range)

    

    # Sort the goal poses by distance from the start pose
    goal_poses.sort(key=lambda goal: calculate_distance(start_pose, goal))
    # Print the distances
    print("Distances from start pose to each goal pose:")
    for i, distance in enumerate(dist):
        print(f"Goal {i+1}: {distance}")
    # Call the service with each sorted goal pose
    for goal_pose in goal_poses:
        dx = start_pose.pose.position.x - goal_pose.pose.position.x
        dy = start_pose.pose.position.y - goal_pose.pose.position.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        print(distance)
        call_path_planning_service(start_pose, goal_pose)
    
    # Wait a bit to ensure all messages are received
    rospy.sleep(5)

    # Print the elapsed times
    print("Elapsed times for each service call (in ms):")
    for i, elapsed in enumerate(time):
        print(f"Goal {i+1}: {elapsed}")
    
    # Print the successes
    print("Successes for each service call:")
    for i, s in enumerate(success):
        print(f"Goal {i+1}: {s}")

    # Save the results to a CSV file
    save_to_csv('results.csv')
    print("Results have been saved to results.csv")

