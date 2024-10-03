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
import threading
from message_filters import ApproximateTimeSynchronizer, Subscriber
from PIL import Image as img
from tf.transformations import euler_from_quaternion
import csv
# mutex = threading.Lock()

GRID_RESOLUTION = 0.1
CLEARANCE_THRESH = 0.5/GRID_RESOLUTION
GRID_SIZE_IN_M = 6
METRIC_CSV = "/habitat-lab/data/data_metrics/Jun_26/metrics_data.csv"
# myargv = rospy.myargv(argv=sys.argv)
# scene = myargv[1]
# driving = myargv[2]
# if driving=="true" or driving =="True":
#     OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_feb_6/driving/"
# else:
#     OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_feb_6/rl/"
OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_jun_26_2/"
# IMAGE_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/images/"+scene
# print(IMAGE_DIR)
max_num = 0
for foldername in os.listdir(OUT_DIR):
    if foldername[:5] != "demo_":
        continue
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

def transform_pose(transformation, pose_wrt_source):
    pose_wrt_target = tf2_geometry_msgs.do_transform_pose(pose_wrt_source, transformation).pose
    return pose_wrt_target

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
        # self.sub_people = rospy.Subscriber("sim/agent_poses", PoseArray, self.people_callback, queue_size=1)
        # self.sub_robot = rospy.Subscriber("sim/robot_pose", PoseStamped, self.get_robot_pose, queue_size=1)
        self.sub_people = Subscriber("sim/agent_poses", PoseArray)
        self.sub_robot = Subscriber("sim/robot_pose", PoseStamped)
        self.sub_door = rospy.Subscriber("sim/door", MarkerArray, self.get_door_pos, queue_size=1)
        self.sub_robot_goal = rospy.Subscriber("sim/goal", Marker, self.get_robot_goal, queue_size=1)
        self.sub_ep_start = rospy.Subscriber("start_ep", Bool, self.is_start, queue_size=1)
        self.sub_click = rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        # self.sub_goal = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_callback, queue_size=100)
        self.cloud_pub = rospy.Publisher("semantic_cloud", PointCloud2, queue_size=2)
        self.sub_img_res = rospy.Subscriber("img_res", Float64, self.set_img_res, queue_size=1)
        # self.image_sub = rospy.Subscriber("/robot_2_rgb",Image,self.img_callback)
        self.image_sub = Subscriber("/robot_2_rgb",Image)
        self.ts = ApproximateTimeSynchronizer([self.sub_people, self.sub_robot, self.image_sub], 10, 0.1, allow_headerless=False)
        self.ts.registerCallback(self.get_data)
        self.reset_sub = rospy.Subscriber("/reload_map_server", Bool, self.reset_callback)
        self.ep_over_sub = rospy.Subscriber("/episode_ended", Bool, self.ep_over_callback)
        self.reset = False
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
        self.grid_size_in_m = GRID_SIZE_IN_M
        self.grid_resolution = GRID_RESOLUTION
        self.img_res = 0.052
        self.grid_dimension = int(self.grid_size_in_m/self.grid_resolution)
        self.grid_img = np.zeros((self.grid_dimension, self.grid_dimension, 3))
        self.overlayed_grid_img = None
        self.update_freq = 1
        self.someone_crossed = False
        self.prev_step = rospy.Time.now()
        self.bad_apples_list = []
        self.prev_robot_traj = None 
        self.prev_human_traj = None
        self.robot_goal = None
        self.ep_counter = 0
        self.metrics = []
        rows = []
        self.current_metric = None
        metric = {}
        with open(METRIC_CSV, mode='r', newline='') as file:
            # Create a DictReader object
            csv_reader = csv.DictReader(file)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Print each row as a dictionary
                rows.append(list(row.values())[0])
        keys = rows[0]
        for row in rows[1:]:
            i =0
            for key in keys:
                metric[key] = row[i]
                i+=1
            self.metrics.append(metric)
            metric = {}
        print("Self metrics is ", self.metrics)
    def reset_callback(self, msg):
        self.reset = msg.data
    
    def ep_over_callback(self, msg):
        
        if msg.data:
            self.ep_counter += 1

    def img_callback(self, msg):
        # rospy.loginfo('Image received')
        # mutex.acquire(blocking=True)
        self.third_rgb_img = self.br.imgmsg_to_cv2(msg)
        
        # self.third_rgb_img[:,:] = [self.third_rgb_img[:,:,2], self.third_rgb_img[:,:,1], self.third_rgb_img[:,:,0]]
        
        for i in np.arange(0,self.grid_size_in_m, self.grid_resolution):
            for j in np.arange(0,self.grid_size_in_m, self.grid_resolution):
                [x,y] = self.grid_to_img([i, j])
                # print("in put and then Point in Image coords is ", [i,j],  [x,y])
                # try:
                    # print("Why ", self.third_rgb_img[y,x])
                self.grid_img[int(np.round(j/self.grid_resolution)), int(np.round(i/self.grid_resolution))] = self.third_rgb_img[y,x]
                # except:
                #     continue
        
        # mutex.release()
        

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
        ans = [int(np.round(point_map_frame[0]/self.img_res)), int(np.round(point_map_frame[1]/self.img_res))]
        return ans
    
    def grid_to_pix(self, point_2d):
        point =  [int(np.round(point_2d[0]/self.grid_resolution)), int(np.round(point_2d[1]/self.grid_resolution))]
        if point[0] <0 or point[1] <0:
            return None
        if point[0] >=self.grid_dimension or point[1]>=self.grid_dimension:
            return None
        return point
    
    def either_clear_door(self):
        try:
            self.robot_cleared = has_cleared_door(self.robot_past_traj[-1], self.robot_past_traj[0], [self.door_start, self.door_end])
            self.human_cleared = has_cleared_door(self.human_past_traj[-1], self.human_past_traj[0], [self.door_start, self.door_end])
            self.someone_crossed = self.robot_cleared or self.human_cleared
        except:
            return False
        return self.someone_crossed

    def set_img_res(self, msg):
        self.img_res = msg.data

    def pub_img_cloud(self):
        points = []
        if self.door_middle is None:
            return
        # if self.overlayed_grid_img is None:
        #     self.overlayed_grid_img = self.grid_img.copy()
        #     print("writing overlayed map ")
        # mutex.acquire(blocking=True)
        pub_stamp = rospy.Time.now()
        for i in np.arange(0,self.grid_size_in_m, self.grid_resolution):
            for j in np.arange(0,self.grid_size_in_m, self.grid_resolution):
                [x,y] = self.grid_to_img([i, j])
                # print("in put and then Point in Image coords is ", [i,j],  [x,y])
                try:
                    # print("Why ", self.third_rgb_img[y,x])
                    [r,g,b] = self.third_rgb_img[y,x]
                    # if [int(np.round(i/self.grid_resolution)), int(np.round(j/self.grid_resolution))] in self.robot_past_traj:
                    #     [r,g,b] = [255,0,0]
                    # if [int(np.round(i/self.grid_resolution)), int(np.round(j/self.grid_resolution))] in self.human_past_traj:
                    #     [r,g,b] = [0,255,0]
                    a = 255
                    z = 0.1
                    rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                    pt = [i, j, z, rgb]
                    points.append(pt)
                except:
                    continue
        print("No. of valid points is ", len(points))
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('rgb', 12, PointField.UINT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
        ]
        
        header = Header()
        header.frame_id = "small_grid_frame"
        pc2 = point_cloud2.create_cloud(header, fields, points)
        pc2.header.stamp = rospy.Time.now()
        self.cloud_pub.publish(pc2)

        robot_msg = Path()
        robot_msg.header.frame_id = "small_grid_frame"
        robot_msg.header.stamp = rospy.Time.now()
        for xy in traj_interp(np.array(self.robot_past_traj)):
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "small_grid_frame"
            pose.pose.position.x = xy[0]*self.grid_resolution
            pose.pose.position.y = xy[1]*self.grid_resolution
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            robot_msg.poses.append(pose)
        self._pub_robot_path.publish(robot_msg)

        human_msg = Path()
        human_msg.header.frame_id = "small_grid_frame"
        human_msg.header.stamp = rospy.Time.now()
        for xy in traj_interp(np.array(self.human_past_traj)):
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = "small_grid_frame"
            pose.pose.position.x = xy[0]*self.grid_resolution
            pose.pose.position.y = xy[1]*self.grid_resolution
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            human_msg.poses.append(pose)
        self._pub_human_path.publish(human_msg)
        # mutex.release()

        


        
    def get_robot_goal(self, msg):
        goal = self.map_to_grid([msg.pose.position.x, msg.pose.position.y])
        self.robot_goal = self.grid_to_pix(goal)





    def get_robot_pose(self, msg):
        # if (self.end_point):
        #     return True
        # rospy.loginfo('Pose received at ')
        # mutex.acquire(blocking=True)
        robot_pose_grid = transform_pose(self.transformation, msg)
        orientation = [robot_pose_grid.orientation.x, robot_pose_grid.orientation.y, robot_pose_grid.orientation.z, robot_pose_grid.orientation.w]
        self.robot_angle = -euler_from_quaternion(orientation)[2]
        if (self.start_point == False):

            self.last_pose_time_stamp = rospy.Time.now()
            robot_pos_map = [msg.pose.position.x, msg.pose.position.y]
            
            robot_pos_grid = self.map_to_grid(robot_pos_map)
            self.robot_pose_2d = self.grid_to_pix(robot_pos_grid)
            self.start_point = True
            if (self.robot_pose_2d) is not None:
                # self.overlayed_grid_img[self.robot_pose_2d[1], self.robot_pose_2d[0]] = [255,0,0]
                self.robot_past_traj.append(self.robot_pose_2d)
            else:
                print("robot not in frame anymore", self.robot_pose_2d)
                pass
        else:

            robot_pos_map = [msg.pose.position.x, msg.pose.position.y]
            robot_pos_grid = self.map_to_grid(robot_pos_map)
            self.robot_pose_2d = self.grid_to_pix(robot_pos_grid)
            
            if (self.robot_pose_2d) is not None:
                # self.overlayed_grid_img[self.robot_pose_2d[1], self.robot_pose_2d[0]] = [255,0,0]
                if not self.robot_pose_2d == self.robot_past_traj[-1]:
                    self.robot_past_traj.append(self.robot_pose_2d)
                    self.last_pose_time_stamp = rospy.Time.now()
            else:
                print("robot not in frame anymore")
                pass
        # mutex.release()
    

                
    def get_door_pos(self, msg):
        array = msg.markers
        door_start = self.map_to_grid([array[0].pose.position.x, array[0].pose.position.y])
        self.door_start = self.grid_to_pix(door_start)
        door_end = self.map_to_grid([array[1].pose.position.x, array[1].pose.position.y])
        self.door_end = self.grid_to_pix(door_end)
        # door_middle = (door_start+door_end)/2
        if self.door_end is not None and self.door_start is not None:
            self.door_middle = self.map_to_grid((np.array(self.door_start)+np.array(self.door_end))/2)
                
    def is_start(self, data):
        self.episode_start = data.data
        self.counter = 0

    def traj_callback(self,data):
        self.traj_feature = [[cell] for cell in data.data]

    def people_callback(self,msg):
        # mutex.acquire(blocking=True)
        human_pos_map = [msg.poses[0].position.x, msg.poses[0].position.y]
        human_pos_grid = self.map_to_grid(human_pos_map)
        self.human_pose_2d = self.grid_to_pix(human_pos_grid)
        a = PoseStamped()
        a.header = msg.header
        a.pose = msg.poses[0]
        human_pose_grid = transform_pose(self.transformation, a)
        orientation = [human_pose_grid.orientation.x, human_pose_grid.orientation.y, human_pose_grid.orientation.z, human_pose_grid.orientation.w]
        self.human_angle = -euler_from_quaternion(orientation)[2]
        if self.human_pose_2d is not None:
            # self.overlayed_grid_img[self.human_pose_2d[1], self.human_pose_2d[0]] = [0,255,0]
            if (self.human_pose_2d not in self.human_past_traj):
                self.human_past_traj.append(self.human_pose_2d)
            else:
                if not (self.human_pose_2d == self.human_past_traj[-1]):
                    self.human_past_traj.append(self.human_pose_2d)
        else:
            print("Human not in grid anymore")
            pass
        # mutex.release()

    def get_data(self, *msg):
        print("No data? ", len(msg))
        print(self.img_res)
        if self.img_res is None:
            return
        self.people_callback(msg[0])
        self.get_robot_pose(msg[1])
        self.img_callback(msg[2])
        # self.pub_img_cloud()
    
    def save_feature(self):
        self.end_point = True
        with open(self.full_path+ "/traj.npy", 'wb') as f:
            np.save(f, np.array(self.robot_past_traj))
        with open(self.full_path+"/rank.txt", 'w') as f:
            f.write('5')
            f.close()
        with open(self.full_path + "/ep_count.txt", 'w') as f:
            f.write(str(self.ep_counter))
            f.close()
        try:
            im = img.fromarray(np.uint8(self.new_overlayed_grid_image))
            im.save(self.full_path+"/final_overlayed_map.png")
            # cv2.imwrite(self.full_path+"/final_overlayed_map.png", self.new_overlayed_grid_image)
        except:
            print("Probably not good demo ", self.full_path)
        metrics = self.metrics[self.ep_counter+1]
        with open(self.full_path+"/metrics.csv", "w", newline="") as fp:
        # Create a writer object
            writer = csv.DictWriter(fp, fieldnames=self.metrics[0])
            writer.writeheader()
            writer.writerow(metrics)
        max_num = 0
        for foldername in os.listdir(OUT_DIR):
            if foldername[:5] != "demo_":
                continue
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
        self.img_res = 0.052

    def get_current_feature(self):
        # self.goal_sink = self.get_goal_sink_feature()
        # print("Saving feature")
        time_between = self.prev_step-rospy.Time.now()
        print("Time between saving is ", time_between.to_sec())
        self.prev_step = rospy.Time.now()
        if self.prev_robot_traj  is not None and self.prev_human_traj is not None:
            if len(self.prev_robot_traj) ==  len(self.robot_past_traj):
                if len(self.prev_human_traj) == len(self.human_past_traj):
                    # embed()
                    # print(self.prev_human_traj)
                    # print(self.human_past_traj)
                    # print("Return no now stuff", len(self.prev_human_traj), len(self.human_past_traj))
                    return
        self.prev_human_traj = self.human_past_traj.copy()
        self.prev_robot_traj = self.robot_past_traj.copy()
        folder_path = self.full_path+"/"+str(self.counter)
        if (self.grid_img == np.zeros(self.grid_img.shape)).all():
            self.bad_apples_list.append(folder_path)
            return
        __ = os.system("mkdir " + self.full_path+"/"+str(self.counter))
        self.new_overlayed_grid_image = self.grid_img.copy()
        
        for robot in self.robot_past_traj:
            self.new_overlayed_grid_image[robot[1], robot[0]] = [255,0,0]
        for human in self.human_past_traj:
            self.new_overlayed_grid_image[human[1], human[0]] = [0,255,0]
        # cv2.imwrite(folder_path+ "/grid_map.png",self.grid_img)
        
        im = img.fromarray(np.uint8(self.grid_img))
        im.save(folder_path+ "/grid_map.png")
        im = img.fromarray(np.uint8(self.new_overlayed_grid_image))
        im.save(folder_path+ "/new_overlayed_grid_map.png")
        # cv2.imwrite(folder_path+ "/new_overlayed_grid_map.png",self.new_overlayed_grid_image)
        with open(folder_path+ "/robot_past_traj.npy", 'wb') as f:
            np.save(f, np.array(self.robot_past_traj))
        with open(folder_path+ "/human_past_traj.npy", 'wb') as f:
            np.save(f, np.array(self.human_past_traj))
        with open (folder_path+ "/heading.npy", 'wb') as f:
            np.save(f, np.array([self.robot_angle, self.human_angle]))
        with open(folder_path+ "/goal.npy", 'wb') as f:
            np.save(f, np.array(self.robot_goal))
        # cv2.imwrite(folder_path+ "/overlayed_grid_map.png",self.overlayed_grid_img)
        im = img.fromarray(np.uint8(self.third_rgb_img))
        im.save(folder_path+ "/raw_img.png")
        # cv2.imwrite(folder_path+ "/raw_img.png", self.third_rgb_img)
        self.grid_img = np.zeros(self.grid_img.shape)
        self.counter += 1
    
    def point_callback(self, data):
        self.episode_start = True
        
    def get_data(self, *msg):
        # print("No data? ", len(msg))
        if (self.img_res is None):
            return
        self.people_callback(msg[0])
        self.get_robot_pose(msg[1])
        self.img_callback(msg[2])
        # self.pub_img_cloud()
        



if __name__ == "__main__":
        rospy.init_node("Feature_expect",anonymous=False)
        # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        feature = FeatureExpect()
        update = 0
        one_cleared = False
        rospy.sleep(1)
        ### Collecting real-time data ###
        
        while(not rospy.is_shutdown()):
            rospy.sleep(0.1)
               
            if feature.reset:
                print("Reset called")
                feature.save_feature()
                __ = os.system("rosnode kill Feature_expect")
                __ = os.system("rosrun habitat_interface get_feat_synch.py")
            if not feature.episode_start:
                continue 
            if feature.door_start is None:
                continue
            if len(feature.robot_past_traj) <1 or len(feature.human_past_traj) <1:
                continue  
            
            # if feature.last_pose_time_stamp is not None:
            #     print("Time is ", (rospy.Time.now()-feature.last_pose_time_stamp).to_sec())
            #     if (rospy.Time.now()-feature.last_pose_time_stamp).to_sec() >8:
            #         print("Time up")
            #         feature.save_feature()

              
            
            
            feature.get_current_feature()
            
                
            if len(feature.human_past_traj) <2:
                continue
            if feature.door_start is None:
                continue
            # print("Check pls", feature.human_past_traj, feature.door_start, feature.door_end)
            human_cleared = has_cleared_door(feature.human_past_traj[-1], feature.human_past_traj[0], [feature.door_start, feature.door_end])
            if (human_cleared and not one_cleared):
                print("Human has cleared the door, saving ")
                feature.save_feature()
                one_cleared = True
                feature.episode_start = True
            # if(feature.either_clear_door() and not one_cleared):
            #     print("Either robot or human has cleared door, saving the full traj need to post fix it here maybe")        
            #     feature.save_feature()
            #     one_cleared = True
            #     feature.episode_start = True
            # if len(feature.robot_past_traj) <2:
            #     continue
            # robot_cleared = has_cleared_door(feature.robot_past_traj[-1], feature.robot_past_traj[0], [feature.door_start, feature.door_end])
            
            # feature.either_clear_door()
            # if (feature.robot_cleared):
            #     print("Robot has cleared the door")
            
            # if (one_cleared and feature.robot_cleared and feature.human_cleared):
            #     print("Saved because both crossed")
            #     feature.save_feature()
            #     feature.episode_start = False
            #     while not feature.reset:
            #         continue
            #     __ = os.system("rosnode kill Feature_expect")
            #     __ = os.system("rosrun habitat_interface get_feat_synch.py")
            
        print(feature.bad_apples_list)


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
