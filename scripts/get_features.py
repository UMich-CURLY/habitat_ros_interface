#!/usr/bin/env python
# note need to run viewer with python2!!!

from cmath import e
import os
import sys
from pyparsing import empty
# from laser2density import Laser2density
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib import colors, markers
# import img_utils
import tf
from tf.transformations import quaternion_matrix
import tf2_ros
import tf2_geometry_msgs
from collections import namedtuple
from threading import Thread
from IPython import embed
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import yaml
import cv2 
def sem_img_to_world(proj, cam, W,H, u, v, debug = False):
    K = proj
    T_world_camera = cam
    rotation_0 = T_world_camera[0:3,0:3]
    translation_0 = T_world_camera[0:3,3]
    uv_1=np.array([[u,v,1]], dtype=np.float32)
    uv_1=np.array([[2*u/W -1,-2*v/H +1,1]], dtype=np.float32)
    uv_1=np.array([[2*v/H -1,-2*u/W +1,1]], dtype=np.float32)
    uv_1=uv_1.T
    assert(W == H)
    if (debug):
        embed()
    inv_rot = np.linalg.inv(rotation_0)
    A = np.matmul(np.linalg.inv(K[0:3,0:3]), uv_1)
    A[2] = 1
    t = np.array([translation_0])
    c = (A-t.T)
    d = inv_rot.dot(c)
    return d

def transform_pose(input_pose, from_frame, to_frame):

    # **Assuming /tf2 topic is being broadcasted
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = input_pose
    pose_stamped.header.frame_id = from_frame
    pose_stamped.header.stamp = rospy.Time.now()

    try:
        # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
        output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(4.0))
        return output_pose_stamped

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        raise

IMAGE_DIR = "/home/catkin_ws/src/habitat_ros_interface/images/current_scene"

Step = namedtuple('Step','cur_state next_state')
class FeatureExpect():
    def __init__(self, gridsize=(3,3), resolution=1):
        self.gridsize = gridsize
        self.resolution = resolution
        # self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        ### Replace with esfm
        self.sub_people = rospy.Subscriber("sim/agent_poses", PoseArray, self.people_callback, queue_size=100)
        # self.sub_goal = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=100)
        self.robot_pose = [0.0, 0.0]
        self.previous_robot_pose = []
        self.robot_pose_rb = [0.0, 0.0]
        self.robot_distance = 0.0
        self.position_offset = [0.0,0.0]
        with open(IMAGE_DIR+"/image_config.yaml", "r") as stream:
            try:
                image_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                raise
        semantic_scene = self.env.sim.semantic_annotations()
        self.chosen_object = semantic_scene.objects[image_config["object_id"]]
        self.semantic_img_H = image_config["H"]
        self.semantic_img_W = image_config["W"]
        with open(image_config["projection_matrix"], 'rb') as f:
            self.semantic_img_proj_mat = np.load(f)
        with open(image_config["camera_matrix"], 'rb') as f:
            self.semantic_img_camera_mat = np.load(f)
        self.semantic_img = cv2.imread(IMAGE_DIR+"/semantic_img.png")   

    def get_robot_pose(self):
        self.tf_listener.waitForTransform("/my_map_frame", "/base_link", rospy.Time(), rospy.Duration(4.0))
        (trans,rot) = self.tf_listener.lookupTransform('/my_map_frame', '/base_link', rospy.Time(0))
        self.robot_pose = [trans[0], trans[1]]
        if(len(self.previous_robot_pose) == 0):
            self.previous_robot_pose = self.robot_pose
        else:
            self.robot_distance += np.sqrt((self.robot_pose[0] - self.previous_robot_pose[0])**2 + (self.robot_pose[1] - self.previous_robot_pose[1])**2)
            self.previous_robot_pose = self.robot_pose
        tf_matrix = quaternion_matrix(rot)
        tf_matrix[0][3] = trans[0]
        tf_matrix[1][3] = trans[1]
        tf_matrix[2][3] = trans[2]
        # print(tf_matrix)
        return tf_matrix

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]

    def people_callback(self,data):
            # print(percent_change)
        agent_poses = data.poses
        # people_stamped = np.array([transform_pose(people, "my_map_frame", "map") for people in agent_poses])
        self.pose_people = np.array([[people.position.x,people.position.y, people.position.z, people.orientation.x, people.orientation.y, people.orientation.z, people.orientation.w] for people in agent_poses])
        self.pose_people_tf = np.empty((0,4 ,4), float)
        for people_pose in self.pose_people:
            rot = people_pose[3:]
            pose_people_tf = quaternion_matrix(rot)
            pose_people_tf[0][3] = people_pose[0]
            pose_people_tf[1][3] = people_pose[1]
            pose_people_tf[2][3] = people_pose[2]
            self.pose_people_tf = np.append(self.pose_people_tf, np.array([pose_people_tf]), axis=0)
    def goal_callback(self,data):
        self.goal = data
        self.received_goal = True
        print("Goal Received")
        

    def get_current_feature(self):
        # 
        # self.localcost_feature = self.Laser2density.temp_result
        self.social_distance_feature = np.ndarray.tolist(self.SocialDistance.get_features())
        # feature_list = [self.social_distance_feature]
        # self.current_feature = np.array([self.distance_feature[i] + self.localcost_feature[i] + self.traj_feature[i] + [0.0] for i in range(len(self.distance_feature))])
        if (self.received_goal):
            self.distance_feature = self.Distance2goal.get_feature_matrix(self.goal)
            self.current_feature = np.array([self.distance_feature[i] + self.traj_feature[i] + [0.0] for i in range(len(self.distance_feature))])
            self.feature_maps.append(np.array(self.current_feature).T)

    def get_expect(self):
        R1 = self.get_robot_pose()

        self.get_current_feature()

        self.feature_expect = np.array([0 for i in range(len(self.current_feature[0]))], dtype=np.float64)

        self.robot_pose_rb = [0.0,0.0]
        
        index = self.in_which_cell(self.robot_pose_rb)
        percent_temp = 0
        while(index):
            # Robot pose
            R2 = self.get_robot_pose()
            
            R = np.dot(np.linalg.inv(R1), R2)

            self.robot_pose_rb = np.dot(R, np.array([[0, 0, 0, 1]]).T)

            self.robot_pose_rb = [self.robot_pose_rb[0][0], self.robot_pose_rb[1][0]]

            index = self.in_which_cell(self.robot_pose_rb)
            print("Relative robot_pose is ", self.robot_pose_rb, "Index is ", index)
            distance = np.sqrt((self.robot_pose[0] - self.goal.pose.position.x)**2+(self.robot_pose[1] - self.goal.pose.position.y)**2)
            if(distance < 0.1 or not index):
                break
            if(not index in self.trajectory):
                self.trajectory.append(index)
            
            # Whether the robot reaches the goal
            
            # print("distance: ", distance)
            step_list = []
            
            rospy.sleep(0.1)

        self.traj = [self.trajectory[i][1]*self.gridsize[1]+self.trajectory[i][0] for i in range(len(self.trajectory))]

        if(len(self.traj) > 1):
            self.trajs.append(np.array(self.traj))
        
        discount = [(1/e)**i for i in range(len(self.trajectory))]
        # for i in range(len(discount)):

        #     self.feature_expect += np.dot(self.current_feature[int(self.trajectory[i][1] * self.gridsize[1] + self.trajectory[i][0])], discount[i])
        
        self.trajectory = []

        # num_changes = abs(sum(self.percent_reward)) / self.robot_distance

        # print("Normalized sudden change: ", num_changes)
	
        # print(self.feature_maps)

    def rot2eul(self, R) :

        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])

        singular = sy < 1e-6

        if not singular :
            z = np.arctan2(R[1,0], R[0,0])
        else :
            z = 0

        return z

    # def reset_robot(self):
    #     self.initpose_pub.publish(self.initpose)
        # print("Publish successfully")

        



if __name__ == "__main__":
        rospy.init_node("Feature_expect",anonymous=False)
        # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        feature = FeatureExpect(resolution=0.1, gridsize=(11,11))

        fm_file = "../dataset/fm/fm.npz"
        traj_file = "../dataset/trajs/trajs.npz"
        # while(not feature.initpose_get):
        #     rospy.sleep(0.1)
        # feature.reset_robot()
        rospy.sleep(1)
        while(not feature.received_goal):
            rospy.sleep(0.1)
        feature.get_current_feature()
        np.savez(fm_file, *feature.feature_maps)
        print("Feature map is ", feature.feature_maps)
        print("Rospy shutdown", rospy.is_shutdown())
        while(not rospy.is_shutdown()):
            feature.get_expect()
            print("Traj is ", feature.trajs)
            if(len(feature.traj) > 1):
                np.savez(traj_file, *feature.trajs)
                print("One demonstration finished!!")
            rospy.sleep(0.1)