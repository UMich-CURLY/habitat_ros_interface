#!/usr/bin/env python
# note need to run viewer with python2!!!

from cmath import e
import os
import sys
from pyparsing import empty
# from laser2density import Laser2density
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, PointStamped, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from matplotlib import colors, markers
# import img_utils
import struct
import tf
from tf.transformations import quaternion_matrix
import tf2_ros
import tf2_geometry_msgs
from collections import namedtuple
from threading import Thread
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import yaml
import cv2 
from IPython import embed
from std_msgs.msg import Bool, Float64
import rosbag
from visualization_msgs.msg import Marker, MarkerArray
CLEARANCE_THRESH = 0.5/0.025
# myargv = rospy.myargv(argv=sys.argv)
# scene = myargv[1]
# driving = myargv[2]
# if driving=="true" or driving =="True":
#     OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_feb_6/driving/"
# else:
#     OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_feb_6/rl/"
OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_mar_26/"
# IMAGE_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/images/"+scene
# print(IMAGE_DIR)
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
# with open("/home/catkin_ws/src/habitat_ros_interface/configs/tasks/pointnav_mp3d.yaml", "r") as stream:
#     try:
#         sim_config = yaml.safe_load(stream)
#     except yaml.YAMLError as exc:
#         print(exc)
#         raise
# episode_path = sim_config["DATASET"]["DATA_PATH"]
# __ = os.system("cp " + episode_path + " " + FULL_PATH)

def transform_point(transformation, point_wrt_source):
    point_wrt_target = \
        tf2_geometry_msgs.do_transform_point(PointStamped(point=point_wrt_source),
            transformation).point
    return [point_wrt_target.x, point_wrt_target.y]

def get_transformation(source_frame, target_frame,
                       tf_cache_duration=2.0):
    tf_buffer = tf2_ros.Buffer(rospy.Duration(tf_cache_duration))
    tf2_ros.TransformListener(tf_buffer)
    transformation = None

    while transformation is None:
    # get the tf at first available time
        try:
            transformation = tf_buffer.lookup_transform(target_frame,
                    source_frame, rospy.Time(0), rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            
            rospy.logerr('Unable to find the transformation from %s to %s', source_frame, target_frame)
            pass
    return transformation

def get_dist_from_door(current_pos, door_pos):
    p2 = np.array(door_pos[1])
    p1 = np.array(door_pos[0])
    p3 = np.array(current_pos)
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

def has_passed_door(current_pos, start_pos, door_pos):
    x2, y2 = door_pos[1]
    x1, y1 = door_pos[0]
    xA, yA = current_pos
    xB, yB = start_pos
    v1 = (x2-x1, y2-y1)   # Vector 1
    v2 = (x2-xA, y2-yA)   # Vector 2
    v3 = (x2-xB, y2-yB)   # Vector 3
    xpA = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product (magnitude)
    xpB = v1[0]*v3[1] - v1[1]*v3[0]  # Cross product (magnitude)
    if xpA > 0 and xpB >0:
        return False
    elif xpA < 0 and xpB < 0:
        return False
    else:
        return True
        
def has_cleared_door(current_pos, start_pos, door_pos):
    dist = get_dist_from_door(current_pos, door_pos) 
    return has_passed_door(current_pos, start_pos, door_pos) and dist > CLEARANCE_THRESH

# Step = namedtuple('Step','cur_state next_state')
class FeatureExpect():
    def __init__(self, gridsize=(3,3), resolution=1):
        
        # self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)
        self.transformation = get_transformation('my_map_frame', 'small_grid_frame')
        self.inv_transform = get_transformation('small_grid_frame', 'my_map_frame')
        self.img_transformation = get_transformation('camera_frame', 'small_grid_frame')
        self.inv_img_transform = get_transformation('small_grid_frame', 'camera_frame')
        print("Got all transforms saved!")
        ### Replace with esfm
        self.sub_people = rospy.Subscriber("sim/agent_poses", PoseArray, self.people_callback, queue_size=1)
        self.sub_robot = rospy.Subscriber("sim/robot_pose", PoseStamped, self.get_robot_pose, queue_size=1)
        self.sub_door = rospy.Subscriber("sim/door", MarkerArray, self.get_door_pos, queue_size=1)
        self.sub_ep_start = rospy.Subscriber("start_ep", Bool, self.is_start, queue_size=1)
        self.sub_click = rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        # self.sub_goal = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=100)
        self.cloud_pub = rospy.Publisher("semantic_cloud", PointCloud2, queue_size=2)
        self.sub_img_res = rospy.Subscriber("img_res", Float64, self.set_img_res, queue_size=1)
        self.image_sub = rospy.Subscriber("/robot_2_rgb",Image,self.img_callback)
        self.br = CvBridge()
        self.third_rgb_img = cv2.imread("/home/catkin_ws/src/habitat_ros_interface/maps/sample_img.png")
        self.map_img = cv2.imread("/home/catkin_ws/src/habitat_ros_interface/maps/sample_map.pgm")
        self.full_path = FULL_PATH
        self.robot_pose = [0.0, 0.0]
        self.previous_robot_pose = []
        self.robot_pose_rb = [0.0, 0.0]
        self.robot_distance = 0.0
        self.position_offset = [0.0,0.0]
        self.robot_past_traj = []
        self.human_past_traj = []
        self.start_point = False
        self.end_point = False
        self.update_num = 0
        self.ep_goal_band = []
        self.human_pose_2d = None
        self.robot_pose_2d = None
        self.door_start = None
        self.door_end = None 
        self.door_middle = None 
        self.last_pose_time_stamp = None
        self.episode_start = False
        self.counter = 0
        self.grid_size_in_m = 4
        self.grid_resolution = 0.05
        self.img_res = None 
        self.grid_dimension = int(self.grid_size_in_m/self.grid_resolution)
        self.grid_img = np.zeros((self.grid_dimension, self.grid_dimension, 3))
        self.overlayed_grid_img = None
        self.update_freq = 1
        self.someone_crossed = False
        
    def img_callback(self, msg):
        rospy.loginfo('Image received...')
        self.third_rgb_img = self.br.imgmsg_to_cv2(msg)
        self.pub_img_cloud()

    def map_to_grid(self, point_2d):
        point = Point(point_2d[0], point_2d[1], 0.0)
        point_grid_frame = transform_point(self.transformation, point)
        return point_grid_frame
    
    def grid_to_map(self, point_2d):
        point = Point(point_2d[0], point_2d[1], 0.0)
        point_map_frame = transform_point(self.inv_transform, point)
        return point_map_frame[:2]
    
    def img_to_grid(self, point_2d):
        point = Point(point_2d[0]/self.img_res, point_2d[1]/self.img_res, 0.0)
        point_grid_frame = transform_point(self.img_transformation, point)
        return point_grid_frame
    
    def grid_to_img(self, point_2d):
        point = Point(point_2d[0], point_2d[1], 0.0)
        point_map_frame = transform_point(self.inv_img_transform, point)
        ans = [int(np.floor(point_map_frame[0]/self.img_res)), int(np.floor(point_map_frame[1]/self.img_res))]
        return ans
    
    def grid_to_pix(self, point_2d):
        return [int(np.floor(point_2d[0]/self.grid_resolution)), int(np.floor(point_2d[1]/self.grid_resolution))]
    
    def either_clear_door(self):
        self.robot_cleared = has_cleared_door(self.robot_past_traj[-1], self.robot_past_traj[0], [self.door_start, self.door_end])
        self.human_cleared = has_cleared_door(self.human_past_traj[-1], self.human_past_traj[0], [self.door_start, self.door_end])
        self.someone_crossed = self.robot_cleared or self.human_cleared
        return self.someone_crossed

    def pub_grid_map(self):
        points = []
        for i in np.arange(0,self.grid_size_in_m, self.grid_resolution):
            for j in np.arange(0,self.grid_size_in_m, self.grid_resolution):
                [x,y] = self.grid_to_map([i, j])
                try:
                    [r, g, b] = self.map_img[int(np.floor((y+1)/0.025)), int(np.floor((x+1)/0.025)), :]
                    [u,v] = self.grid_to_pix([i,j])
                    self.grid_img[u,v, :] = self.map_img[int(np.floor((y+1)/0.025)), int(np.floor((x+1)/0.025)), :]
                    a = 255
                    z = 0.1
                    rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                    pt = [x, y, z, rgb]
                    points.append(pt)
                except:
                    continue

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('rgb', 12, PointField.UINT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]
        header = Header()
        header.frame_id = "my_map_frame"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.cloud_pub.publish(pc2)

    def set_img_res(self, msg):
        self.img_res = msg.data

    def pub_img_cloud(self):
        points = []
        if self.door_middle is None:
            return
        # self.third_rgb_img = cv2.imread("/home/catkin_ws/src/habitat_ros_interface/maps/sample_img.png")
        for i in np.arange(0,self.grid_size_in_m, self.grid_resolution):
            for j in np.arange(0,self.grid_size_in_m, self.grid_resolution):
                [x,y] = self.grid_to_img([i, j])
                # print("in put and then Point in Image coords is ", [i,j],  [x,y])
                try:
                    # print("Why ", self.third_rgb_img[y,x])
                    [r,g,b] = self.third_rgb_img[y,x]
                    self.grid_img[int(np.floor(i/self.grid_resolution)), int(np.floor(j/self.grid_resolution))] = self.third_rgb_img[x,y]
                    # [r,g,b] = [255,0,0]
                    a = 255
                    z = 0.1
                    rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                    pt = [i, j, z, rgb]
                    points.append(pt)
                except:
                    continue
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('rgb', 12, PointField.UINT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]
        if self.overlayed_grid_img is None:
            self.overlayed_grid_img = self.grid_img.copy()
            print("writing overlayed map ")
        header = Header()
        header.frame_id = "small_grid_frame"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.cloud_pub.publish(pc2)

    def pub_grid_map_test(self):
        points = []
        if self.door_middle is None:
            return
        
        for i in np.arange(0,self.grid_size_in_m, self.grid_resolution):
            for j in np.arange(0,self.grid_size_in_m, self.grid_resolution):
                [x,y] = self.grid_to_map([i, j])
                try:
                    on_same_side = has_cleared_door([i/self.grid_resolution,j/self.grid_resolution], [1,1], [self.door_start, self.door_end])
                    if on_same_side:
                        [r, g, b] = [255,0,0]
                    else:
                        [r, g, b] = [0, 255, 0]
                    [u,v] = self.grid_to_pix([i,j])
                    self.grid_img[u,v, :] = self.map_img[int(np.floor((y+1)/0.025)), int(np.floor((x+1)/0.025)), :]
                    a = 255
                    z = 0.1
                    rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                    pt = [x, y, z, rgb]
                    points.append(pt)
                except:
                    continue
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('rgb', 12, PointField.UINT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
        ]
        header = Header()
        header.frame_id = "my_map_frame"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.cloud_pub.publish(pc2)


    def get_robot_pose(self, msg):
        if (self.end_point):
            return True
        if (self.start_point == False and self.overlayed_grid_img is not None):
            self.last_pose_time_stamp = rospy.Time.now()
            robot_pos_map = [msg.pose.position.x, msg.pose.position.y]
            
            robot_pos_grid = self.map_to_grid(robot_pos_map)
            self.robot_pose_2d = self.grid_to_pix(robot_pos_grid)
            self.start_point = True
            try:
                self.overlayed_grid_img[self.robot_pose_2d[1], self.robot_pose_2d[0]] = [255,0,0]
                if self.robot_pose_2d not in self.robot_past_traj:
                    self.robot_past_traj.append(self.robot_pose_2d)
            except:
                print("robot not in frame anymore", self.robot_pose_2d)
                pass
        else:

            robot_pos_map = [msg.pose.position.x, msg.pose.position.y]
            robot_pos_grid = self.map_to_grid(robot_pos_map)
            self.robot_pose_2d = self.grid_to_pix(robot_pos_grid)
            
            try:
                self.overlayed_grid_img[self.robot_pose_2d[1], self.robot_pose_2d[0]] = [255,0,0]
                if self.robot_pose_2d not in self.robot_past_traj:
                    self.robot_past_traj.append(self.robot_pose_2d)
                    self.last_pose_time_stamp = rospy.Time.now()
                    
            except:
                # print("robot not in frame anymore")
                pass
    

                
    def get_door_pos(self, msg):
        array = msg.markers
        door_start = self.map_to_grid([array[0].pose.position.x, array[0].pose.position.y])
        self.door_start = self.grid_to_pix(door_start)
        door_end = self.map_to_grid([array[1].pose.position.x, array[1].pose.position.y])
        self.door_end = self.grid_to_pix(door_end)
        # door_middle = (door_start+door_end)/2
        self.door_middle = self.map_to_grid((np.array(self.door_start)+np.array(self.door_end))/2)
                
    def is_start(self, data):
        self.episode_start = data.data
        self.counter = 0

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]

    def people_callback(self,msg):
            # print(percent_change)
        
        human_pos_map = [msg.poses[0].position.x, msg.poses[0].position.y]
        human_pos_grid = self.map_to_grid(human_pos_map)
        self.human_pose_2d = self.grid_to_pix(human_pos_grid)
        try:
            self.overlayed_grid_img[self.human_pose_2d[1], self.human_pose_2d[0]] = [0,255,0]
            if (self.human_pose_2d not in self.human_past_traj):
                self.human_past_traj.append(self.human_pose_2d)
        except:
            # print("Human not in grid anymore")
            pass
    
    def save_feature(self):
        self.end_point = True
        with open(self.full_path+ "/traj.npy", 'wb') as f:
            np.save(f, np.array(self.robot_past_traj))
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
        self.full_path = next_folder_name
        self.end_point = False
        self.start_point = False
        self.counter = 0

    def get_current_feature(self):
        # self.goal_sink = self.get_goal_sink_feature()
        print("Saving feature")
        __ = os.system("mkdir " + self.full_path+"/"+str(self.counter))
        self.new_overlayed_grid_image = self.grid_img.copy()
        folder_path = self.full_path+"/"+str(self.counter)
        for robot in self.robot_past_traj:
            self.new_overlayed_grid_image[robot[1], robot[0]] = [255,0,0]

        for human in self.human_past_traj:
            self.new_overlayed_grid_image[human[1], human[0]] = [0,255,0]
        cv2.imwrite(folder_path+ "/grid_map.png",self.grid_img)
        cv2.imwrite(folder_path+ "/new_overlayed_grid_map.png",self.new_overlayed_grid_image)
        with open(folder_path+ "/robot_past_traj.npy", 'wb') as f:
            np.save(f, np.array(self.robot_past_traj))
        print ("whats is the issue ", self.human_past_traj)
        with open(folder_path+ "/human_past_traj.npy", 'wb') as f:
            np.save(f, np.array(self.human_past_traj))
        cv2.imwrite(folder_path+ "/overlayed_grid_map.png",self.overlayed_grid_img)
        self.counter += 1



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
    def point_callback(self, data):
        print("Saving episode till now and creating new one ")
        self.save_feature()
        print("human traj was ", self.human_past_traj)
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
        global FULL_PATH
        FULL_PATH = next_folder_name
        self.robot_traj = []
        self.traj = []
        for pose in self.human_future_traj:
            self.human_past_traj.append(pose)
        print("human traj now is ", self.human_past_traj)
        self.human_future_traj = []
        self.end_point = False
        self.start_point = False

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
        one_cleared = False
        ### Collecting real-time data ###
        while(not rospy.is_shutdown()):
            rospy.sleep(0.1)
            if feature.last_pose_time_stamp is not None:
                print("Time is ", (rospy.Time.now()-feature.last_pose_time_stamp).to_sec())
                if (rospy.Time.now()-feature.last_pose_time_stamp).to_sec() >10:
                    print("Time up")
                    feature.save_feature()
                    exit(0)
            print(feature.robot_past_traj)
            if len(feature.robot_past_traj) == 0:
                continue
            if(feature.either_clear_door() and not one_cleared):
                print("Either robot or human has cleared door, saving the full traj need to post fix it here maybe")        
                feature.save_feature()
                one_cleared = True
            feature.either_clear_door()
            if (one_cleared and feature.robot_cleared and feature.human_cleared):
                feature.save_feature()
                exit(0)
            # feature.map_to_grid([1,1])
            
            feature.get_current_feature()


        ### Collecting data from bag file ### 
        # bag_ended = False
        # bag = rosbag.Bag("/home/catkin_ws/src/habitat_ros_interface/bags/bag1_2024-03-26-00-21-53.bag")
        # end_time = bag.get_end_time()
        
        # while not bag_ended:
        #     feature.pub_grid_map_test()
        #     if (abs(rospy.Time.now().to_sec()-end_time) < 0.1):
        #         bag_ended = True
        #         print("Time up")
        #         feature.save_feature()
        #         exit(0)
