#!/usr/bin/env python
# note need to run viewer with python2!!!

from cmath import e
import os
import sys
from pyparsing import empty
# from laser2density import Laser2density
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, PointStamped
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
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import yaml
import cv2 
from IPython import embed
from std_msgs.msg import Bool, Int32MultiArray, MultiArrayLayout, MultiArrayDimension
myargv = rospy.myargv(argv=sys.argv)
scene = myargv[1]
driving = myargv[2]

IMAGE_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/images/"+scene
print(IMAGE_DIR)
FULL_PATH = "docker_path"
TEMP_PATH = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_jan_12/test"
with open("/home/catkin_ws/src/habitat_ros_interface/configs/tasks/pointnav_mp3d.yaml", "r") as stream:
    try:
        sim_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise
episode_path = sim_config["DATASET"]["DATA_PATH"]
def sem_img_to_world(proj, cam, W,H, u, v, robot_height, debug = False):
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
    A[2] = 0.04
    t = np.array([translation_0])
    c = (A-t.T)
    d = inv_rot.dot(c)
    # print("Using prev method ", d)
    #### Work with new height 
    uv_1 = np.append(uv_1, 1.0)
    A = np.matmul(np.linalg.inv(K[0:4,0:4]), uv_1)
    new_A =np.array([A[0],-A[1], -1.39-robot_height/2, 1.0])
    d = np.matmul(np.linalg.inv(T_world_camera), new_A)
    return d[0:3]

def world_to_sem_img(proj, cam, agent_state, W, H, debug = False):
    K = proj
    T_cam_world = cam
    pos = np.array([agent_state[0], agent_state[1], agent_state[2], 1.0])
    projection = np.matmul(T_cam_world, pos)
    # projection = np.array([projection[0], projection[2], projection[1], 1.0])
    image_coordinate = np.matmul(K, projection)
    if (debug):
        embed()
    image_coordinate = image_coordinate/image_coordinate[2]
    v = H-(image_coordinate[0]+1)*(H/2)
    u = W-(1-image_coordinate[1])*(W/2)
    if u >=W:
        u = W-1
    if u<0:
        u = 0
    if v>=H:
        v = H-1
    if v<0:
        v = 0    
    return [int(u),int(v)]

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


def traj_interp(c):
    d = c.astype(int)
    iter = len(d) - 1
    added = 0
    i = 0
    while i < iter:
        while np.sqrt((d[i+added,0]-d[i+1+added,0])**2 + (d[i+added,1]-d[i+1+added,1])**2) > np.sqrt(1):
            d = np.insert(d, i+added+1, [0, 0], axis=0)
            if d[i+added+2, 0] - d[i+added, 0] > 0:
                d[i+added+1, 0] = d[i+added, 0] + 1
                d[i+added+1, 1] = d[i+added, 1]
            elif d[i+added+2, 0] - d[i+added, 0] < 0:
                d[i+added+1, 0] = d[i+added, 0] - 1
                d[i+added+1, 1] = d[i+added, 1]
            else:
                d[i+added+1, 0] = d[i+added, 0]
                if d[i+added+2, 1] - d[i+added, 1] > 0:
                    d[i+added+1, 1] = d[i+added, 1] + 1
                elif d[i+added+2, 1] - d[i+added, 1] < 0:
                    d[i+added+1, 1] = d[i+added, 1] - 1
                else:
                    d[i+added+1, 1] = d[i+added, 1]
            added += 1
        i += 1
    connected_map = np.zeros((32, 32))
    for i in range(len(d)):
        connected_map[int(d[i,1])+1, int(d[i,0])+1] = 1
    return d

Step = namedtuple('Step','cur_state next_state')
class FeatureExpect():
    def __init__(self, gridsize=(3,3), resolution=1):
        
        # self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        ### Replace with esfm
        self.sub_people = rospy.Subscriber("human_pose_in_sim", Pose, self.people_callback, queue_size=1)
        self.sub_robot = rospy.Subscriber("robot_pose_in_sim", Pose, self.get_robot_pose, queue_size=1)
        self.sub_ep_start = rospy.Subscriber("start_ep", Bool, self.is_start, queue_size=1)
        self._pub_all_agents = rospy.Publisher("~human_traj", Int32MultiArray, queue_size = 1)
        self._pub_robot = rospy.Publisher("~robot_traj", Int32MultiArray, queue_size = 1)
        self.sub_click = rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        # self.sub_goal = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=100)
        self.robot_pose = [0.0, 0.0]
        self.previous_robot_pose = []
        self.robot_pose_rb = [0.0, 0.0]
        self.robot_distance = 0.0
        self.position_offset = [0.0,0.0]
        self.robot_traj = []
        self.human_past_traj = []
        self.human_future_traj = []
        self.traj = []
        self.start_point = False
        self.end_point = False
        self.update_num = 0
        self.ep_goal_band = []
        self.human_pose_2d = None
        self.robot_pose_2d = None
        self.last_pose_time_stamp = None
        self.episode_start = False
        self.counter = 0
        ## Read and publush goal sink feature and semantic image 
        with open(IMAGE_DIR+"/image_config.yaml", "r") as stream:
            try:
                image_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                raise
        self.semantic_img_H = image_config["H"]
        self.semantic_img_W = image_config["W"]
        self.semantic_img_resolution = image_config["resolution"]
        self.gridsize = [self.semantic_img_H, self.semantic_img_W]
        self.resolution = self.semantic_img_resolution
        with open(image_config["projection_matrix"], 'rb') as f:
            self.semantic_img_proj_mat = np.load(f)
        with open(image_config["camera_matrix"], 'rb') as f:
            self.semantic_img_camera_mat = np.load(f)
        self.door_center =  image_config["door_center"]
        with open(image_config["world_to_door"], 'rb') as f:
            self.world_to_door =  np.load(f)
        self.semantic_img = cv2.imread(IMAGE_DIR+"/semantic_img.png") 
        self.goal_sink_img = cv2.imread(IMAGE_DIR+"/goal_sink.png") 
        

    def get_robot_pose(self, msg):
        if (self.end_point):
            return True
        
        if (self.start_point == False):
            with open(IMAGE_DIR+"/image_config.yaml", "r") as stream:
                try:
                    image_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
                    raise
            self.semantic_img_H = image_config["H"]
            self.semantic_img_W = image_config["W"]
            self.semantic_img_resolution = image_config["resolution"]
            self.gridsize = [self.semantic_img_H, self.semantic_img_W]
            self.resolution = self.semantic_img_resolution
            with open(image_config["projection_matrix"], 'rb') as f:
                self.semantic_img_proj_mat = np.load(f)
            with open(image_config["camera_matrix"], 'rb') as f:
                self.semantic_img_camera_mat = np.load(f)
            self.door_center =  image_config["door_center"]
            with open(image_config["world_to_door"], 'rb') as f:
                self.world_to_door =  np.load(f)
            self.semantic_img = cv2.imread(IMAGE_DIR+"/semantic_img.png") 
            robot_pos_3d = [msg.position.x, msg.position.y, msg.position.z]
            self.robot_height = robot_pos_3d[1]
            self.update_num+=1
            robot_pose_2d = world_to_sem_img(self.semantic_img_proj_mat, self.semantic_img_camera_mat, robot_pos_3d, self.semantic_img.shape[0], self.semantic_img.shape[1])
            
            with open("/home/catkin_ws/src/habitat_ros_interface/configs/tasks/pointnav_mp3d.yaml", "r") as stream:
                try:
                    sim_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
                    raise
            episode_path = sim_config["DATASET"]["DATA_PATH"]
            __ = os.system("cp " + episode_path + " " + TEMP_PATH)
            # if (self.is_point_in_band(robot_pose_2d)):
            print("Found the first robot pose")
            self.start_point = True
            self.robot_traj.append(robot_pose_2d)
            if self.human_pose_2d is not None:
                self.traj.append((robot_pose_2d, self.human_pose_2d))
            __ = os.system("cp "+ IMAGE_DIR+"/goal_sink.png " + TEMP_PATH)
            __ = os.system("cp "+ IMAGE_DIR+"/semantic_img.png " + TEMP_PATH)
            self.semantic_img[robot_pose_2d[0], robot_pose_2d[1]] = [0,0,0]
            self.robot_pose_2d = robot_pose_2d
            return robot_pose_2d
        else:

            robot_pos_3d = [msg.position.x, msg.position.y, msg.position.z]
            self.robot_height = robot_pos_3d[1]
            self.update_num+=1
            robot_pose_2d = world_to_sem_img(self.semantic_img_proj_mat, self.semantic_img_camera_mat, robot_pos_3d, self.semantic_img.shape[0], self.semantic_img.shape[1])

            world_coordinates = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1],robot_pose_2d[0],robot_pose_2d[1], self.robot_height)
            self.semantic_img[robot_pose_2d[0], robot_pose_2d[1]] = [0+self.counter,0,0]
            if robot_pose_2d not in self.robot_traj:
                self.robot_traj.append(robot_pose_2d)

            if ((robot_pose_2d, self.human_pose_2d) not in self.traj and self.human_pose_2d is not None):
                self.counter = self.counter+1
                self.traj.append((robot_pose_2d, self.human_pose_2d))
                self.last_pose_time_stamp = rospy.Time.now()
                print("Traj is ", self.traj)
            # if (self.update_num == 1):
            #     test_pos_3d = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], robot_pose_2d[0], robot_pose_2d[1], robot_pos_3d[1], debug = True)
            # test_pos_3d = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], robot_pose_2d[0], robot_pose_2d[1], robot_pos_3d[1])
            
            robot_start_pose = self.traj[0][0]
            robot_start_coord = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], robot_start_pose[0], robot_start_pose[1], self.robot_height)
            # print("Is in band? ", self.is_point_in_band(robot_pose_2d,[0.8,2.5]) )
            # print("Is on other side? ", self.is_point_on_other_side(robot_start_coord, world_coordinates))
            # if(self.is_point_in_band(robot_pose_2d,[0.8,2.5])):
            #     if(self.is_point_on_other_side(robot_start_coord, world_coordinates)):
            #         self.end_point = True
            #         print("saving image", self.traj)
            #         cv2.imwrite(FULL_PATH+ "/traj_feat.png",self.semantic_img)
            #         with open(FULL_PATH+ "/trajectory.npy", 'wb') as f:
            #             np.save(f, np.array(self.traj))
            self.robot_pose_2d = robot_pose_2d
        
            
                
    def is_start(self, data):
        self.episode_start = data.data
        if(self.episode_start):
            self.save_feature()
        self.counter = 0

    def point_callback(self, data):
        print("getting continued data")

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]

    def people_callback(self,msg):
            # print(percent_change)
        
        human_pos_3d = [msg.position.x, msg.position.y, msg.position.z]
        self.human_height = human_pos_3d[1]
        self.update_num+=1
        human_pose_2d = world_to_sem_img(self.semantic_img_proj_mat, self.semantic_img_camera_mat, human_pos_3d, self.semantic_img.shape[0], self.semantic_img.shape[1])

        world_coordinates = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1],human_pose_2d[0],human_pose_2d[1], self.human_height)
        self.semantic_img[human_pose_2d[0], human_pose_2d[1]] = [0,self.counter,0]
        self.human_pose_2d = human_pose_2d
        # if self.episode_start:
        #     if (self.human_pose_2d not in self.human_future_traj):
        #         self.human_future_traj.append(self.human_pose_2d)
        # else:
        if (self.human_pose_2d not in self.human_past_traj):
            self.human_past_traj.append(self.human_pose_2d)
    
    def save_feature(self):
        # self.end_point = True
        print("saving image", self.traj)
        cv2.imwrite(TEMP_PATH+ "/traj_feat.png",self.semantic_img)
        msg = Int32MultiArray()
        data = traj_interp(np.array(self.human_past_traj))
        layout = MultiArrayLayout()
        dim0 = MultiArrayDimension()
        dim1 = MultiArrayDimension()
        dim0.label = "traj"
        dim1.label = "xy"
        dim0.stride = len(data)*2
        dim0.size = len(data)
        dim1.size = 2
        dim1.stride = 2
        layout.dim.append(dim0)
        layout.dim.append(dim1)
        layout.data_offset = 0
        msg.layout = layout
        data = data.reshape([1,len(data)*2])
        msg.data = np.ndarray.tolist(data[0])
        self._pub_all_agents.publish(msg)
        msg_robot = Int32MultiArray()
        layout = MultiArrayLayout()
        dim0 = MultiArrayDimension()
        dim0.label = "traj"
        dim0.stride = 2
        dim0.size = 2
        layout.dim.append(dim0)
        layout.data_offset = 0
        msg_robot.layout = layout
        msg_robot.data = self.robot_pose_2d
        print(msg_robot.data)
        print(data)
        self._pub_robot.publish(msg_robot)

        # with open(TEMP_PATH+ "/trajectory.npy", 'wb') as f:
        #     np.save(f, np.array(self.traj))
        with open(TEMP_PATH+ "/robot_traj.npy", 'wb') as f:
            np.save(f, np.array([self.robot_traj[-1]]))
        with open(TEMP_PATH+ "/human_past_traj.npy", 'wb') as f:
            np.save(f, np.array(self.human_past_traj))
        # with open(FULL_PATH+ "/human_traj.npy", 'wb') as f:
        #     np.save(f, np.array(self.human_future_traj))


    


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
        rospy.init_node("Feature_publish",anonymous=False)
        # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        feature = FeatureExpect()
        update = 0
        while(not rospy.is_shutdown()):
            rospy.sleep(0.01)
            if feature.last_pose_time_stamp is not None:
                if (rospy.Time.now()-feature.last_pose_time_stamp).to_sec() >30:
                    feature.save_feature()
            # print("Traj is ", feature.traj)
            # update +=1
            # print(update)
            # if (update==300):
            #     cv2.imwrite("try_in_feat.png",feature.semantic_img)
            
