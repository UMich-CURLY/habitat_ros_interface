#!/usr/bin/env python
# note need to run viewer with python2!!!

import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import numpy as np


RGB_IMG_WIDTH = 720
RGB_IMG_HEIGHT = 720

rospy.init_node("npthirdrgb2ros_rgb",anonymous=False)

pub_1 = rospy.Publisher("robot_1_third_rgb", Image, queue_size=10)
# camera_info_pub_1 = rospy.Publisher("robot_1_camera_info_topic", CameraInfo, queue_size=0)

def callback_1(data):
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    h.frame_id = "camera_frame"
    img_raveled = data.data[0:-2]
    img_size = data.data[-2:].astype(int)
    
    img = (np.reshape(img_raveled, (img_size[0], img_size[1], 3))).astype(np.uint8)
    image_message = CvBridge().cv2_to_imgmsg(img, encoding="rgb8")
    image_message.header = h
    pub_1.publish(image_message)
    
    camera_info_msg = CameraInfo()
    camera_info_msg.header = h
    fx, fy = RGB_IMG_WIDTH/2, RGB_IMG_HEIGHT/2
    cx, cy = RGB_IMG_WIDTH/2, RGB_IMG_HEIGHT/2

    camera_info_msg.width = RGB_IMG_WIDTH
    camera_info_msg.height = RGB_IMG_HEIGHT
    camera_info_msg.distortion_model = "plumb_bob"
    camera_info_msg.K = np.float32([fx, 0, cx, 0, fy, cy, 0, 0, 1])

    camera_info_msg.D = np.float32([0, 0, 0, 0, 0])

    camera_info_msg.P = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0]

    # camera_info_pub_1.publish(camera_info_msg)

def listener():
    
    rospy.Subscriber("third_rgb_1", numpy_msg(Floats), callback_1)
    rospy.spin()


if __name__ == "__main__":
    listener()
