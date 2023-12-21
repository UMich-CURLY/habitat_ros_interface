#!/usr/bin/env python
# note need to run viewer with python2!!!

from cmath import e
import os
import sys
from pyparsing import empty
# from laser2density import Laser2density
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
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
myargv = rospy.myargv(argv=sys.argv)

scene = myargv[1]
OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl/"
IMAGE_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/images/"+scene
print(IMAGE_DIR)
max_num = 0
for foldername in os.listdir(OUT_DIR):
    number_str = "_"
    valid = False
    m = foldername[5:]
    max_num = max(max_num,int(m))
next_folder_name = OUT_DIR+"demo_"+str(max_num)
entry = os.listdir(next_folder_name)
if (not (len(entry) == 0)):
    next_folder_name = OUT_DIR+"demo_"+str(max_num+1)
    __ = os.system("mkdir " + next_folder_name)
print ("new folder is , continue?", next_folder_name)
FULL_PATH = next_folder_name
with open("/home/catkin_ws/src/habitat_ros_interface/configs/tasks/pointnav_mp3d.yaml", "r") as stream:
    try:
        sim_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise
episode_path = sim_config["DATASET"]["DATA_PATH"]
__ = os.system("cp " + episode_path + " " + FULL_PATH)
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




Step = namedtuple('Step','cur_state next_state')
class FeatureExpect():
    def __init__(self, gridsize=(3,3), resolution=1):
        
        # self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        ### Replace with esfm
        self.sub_people = rospy.Subscriber("human_pose_in_sim", Pose, self.people_callback, queue_size=1)
        self.sub_robot = rospy.Subscriber("robot_pose_in_sim", Pose, self.get_robot_pose, queue_size=1)
        # self.sub_goal = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=100)
        self.robot_pose = [0.0, 0.0]
        self.previous_robot_pose = []
        self.robot_pose_rb = [0.0, 0.0]
        self.robot_distance = 0.0
        self.position_offset = [0.0,0.0]
          
        self.traj = []
        self.start_point = False
        self.end_point = False
        self.update_num = 0
        self.ep_goal_band = []


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
            __ = os.system("cp " + episode_path + " " + FULL_PATH)
            # if (self.is_point_in_band(robot_pose_2d)):
            print("Found the first robot pose")
            self.start_point = True
            self.traj.append(robot_pose_2d)
            self.get_current_feature()
            __ = os.system("cp "+ IMAGE_DIR+"/goal_sink.png " + FULL_PATH)
            __ = os.system("cp "+ IMAGE_DIR+"/semantic_img.png " + FULL_PATH)
            self.semantic_img[robot_pose_2d[0], robot_pose_2d[1]] = [0,0,0]
            return robot_pose_2d
        else:

            robot_pos_3d = [msg.position.x, msg.position.y, msg.position.z]
            self.robot_height = robot_pos_3d[1]
            self.update_num+=1
            robot_pose_2d = world_to_sem_img(self.semantic_img_proj_mat, self.semantic_img_camera_mat, robot_pos_3d, self.semantic_img.shape[0], self.semantic_img.shape[1])

            world_coordinates = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1],robot_pose_2d[0],robot_pose_2d[1], self.robot_height)
            self.semantic_img[robot_pose_2d[0], robot_pose_2d[1]] = [0,0,0]
            if (robot_pose_2d not in self.traj):
                self.traj.append(robot_pose_2d)
                print(robot_pose_2d)
            # if (self.update_num == 1):
            #     test_pos_3d = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], robot_pose_2d[0], robot_pose_2d[1], robot_pos_3d[1], debug = True)
            # test_pos_3d = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], robot_pose_2d[0], robot_pose_2d[1], robot_pos_3d[1])
            
            robot_start_pose = self.traj[0]
            robot_start_coord = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], robot_start_pose[0], robot_start_pose[1], self.robot_height)
            
            # print("Is in band? ", self.is_point_in_band(robot_pose_2d,[0.8,2.5]) )
            # print("Is on other side? ", self.is_point_on_other_side(robot_start_coord, world_coordinates))
            if(self.is_point_in_band(robot_pose_2d,[0.8,2.5])):
                if(self.is_point_on_other_side(robot_start_coord, world_coordinates)):
                    self.end_point = True
                    print("saving image", len(self.traj))
                    cv2.imwrite(FULL_PATH+ "/traj_feat.png",self.semantic_img)
                    with open(FULL_PATH+ "/trajectory.npy", 'wb') as f:
                        np.save(f, np.array(self.traj))
            
                
        

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]

    def people_callback(self,msg):
            # print(percent_change)
        
        human_pos_3d = [msg.position.x, msg.position.y, msg.position.z]
        self.human_height = human_pos_3d[1]
        self.update_num+=1
        human_pose_2d = world_to_sem_img(self.semantic_img_proj_mat, self.semantic_img_camera_mat, human_pos_3d, self.semantic_img.shape[0], self.semantic_img.shape[1])

        world_coordinates = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1],human_pose_2d[0],human_pose_2d[1], self.human_height)
        self.semantic_img[human_pose_2d[0], human_pose_2d[1]] = [255,0,0]
        
    
    def get_people_feature():
        pass

    def get_current_feature(self):
        # self.goal_sink = self.get_goal_sink_feature()
        print("Saving feature")
        # cv2.imwrite(FULL_PATH+ "/goal_sink.png", self.goal_sink)


    def get_goal_sink_feature(self, goal_band = [1.0,1.5]):
        empty_image = 0*np.ones(self.semantic_img.shape)
        robot_start_pose = self.traj[0]
        robot_start_coord = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], robot_start_pose[0], robot_start_pose[1], self.robot_height)
        robot_dist = self.get_dist_from_door(robot_start_pose)
        goal_band[0] = robot_dist - 0.05
        goal_band[1] = robot_dist + 0.05
        self.ep_goal_band = goal_band
        print(goal_band)
        for i in range(0,self.semantic_img.shape[0],1):
            for j in range(0,self.semantic_img.shape[1], 1):
                world_coordinates = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1],i,j, self.robot_height)
                # print("Coords", world_coordinates[2], world_coordinates[0])
                world_coordinates[1] = self.robot_height
                # reverse = world_to_sem_img(self.semantic_img_proj_mat, self.semantic_img_camera_mat, world_coordinates, self.semantic_img.shape[0], self.semantic_img.shape[1])
                # print([i,j], reverse)
                if(self.is_point_in_band([i,j], goal_band)):
                    if(self.is_point_on_other_side(robot_start_coord, world_coordinates)):
                        empty_image[i,j] = [255,0,0]
                    else:
                        empty_image[i,j] = [0,255,0]
        return empty_image

    def is_point_in_band(self, point, goal_band = [1.0,1.5]):
        dist = self.get_dist_from_door(point)
        if (dist >goal_band[0] and dist< goal_band[1]):
            return True
        else:
            return False
    def get_dist_from_door(self,point):
        center_gt = [self.door_center[2], self.door_center[0]]
        world_coordinates = sem_img_to_world(self.semantic_img_proj_mat, self.semantic_img_camera_mat, self.semantic_img.shape[0], self.semantic_img.shape[1], point[0], point[1], self.robot_height)
        [x,y] = [world_coordinates[2], world_coordinates[0]]
        dist = np.linalg.norm(np.array(center_gt)-np.array([x,y]))
        return dist
    
    def get_dist_from_door_3d(self,point3d):
        center_gt = [self.door_center[2], self.door_center[0]]
        [x,y] = [point3d[2], point3d[0]]
        dist = np.linalg.norm(np.array(center_gt)-np.array([x,y]))
        return dist
    
    def is_point_on_other_side(self, p1, p2):
        transform = self.world_to_door
        p1_local = np.matmul(transform, np.append(p1,1.0).T)
        p2_local = np.matmul(transform, np.append(p2,1.0).T)
        y1 = p1_local[2]
        y2 = p2_local[2]
        x1 = p1_local[1]
        x2 = p2_local[1]

        if (np.sign(y1) == np.sign(y2) or abs(y1) <5 or abs(y2)<5):
            return False
        else:
            # print(p1_local, p2_local)
            return True
        


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
        feature = FeatureExpect()
        update = 0
        while(not rospy.is_shutdown()):
            rospy.sleep(0.01)
            # print("Traj is ", feature.traj)
            # update +=1
            # print(update)
            # if (update==300):
            #     cv2.imwrite("try_in_feat.png",feature.semantic_img)
            
