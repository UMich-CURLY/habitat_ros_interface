#!/usr/bin/env python
# note need to run viewer with python2!!!

import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import std_msgs
import cv2
import numpy as np

rospy.init_node("npbc_sensor2ros_rgb",anonymous=False)

pub = rospy.Publisher("img_repub", Image, queue_size=10)


def callback(image_message):
    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    image_message.header = h
    pub.publish(image_message)


def listener():
    
    rospy.Subscriber("/robot_2_rgb", Image, callback)
    rospy.spin()


if __name__ == "__main__":
    listener()
