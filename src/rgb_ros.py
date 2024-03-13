#!/usr/bin/env python
# note need to run viewer with python2!!!
import sys
print(sys.path)
# sys.path = ['', '/home/catkin_ws/devel/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/python2.7/dist-packages/wx-3.0-gtk2']
import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np

rospy.init_node("nprgb2ros_rgb",anonymous=False)

pub_1 = rospy.Publisher("robot_1_rgb", Image, queue_size=10)
pub_2 = rospy.Publisher("robot_2_rgb", Image, queue_size=10)
pub_3 = rospy.Publisher("robot_3_rgb", Image, queue_size=10)

def callback_1(data):
    img_raveled = data.data[0:-2]
    img_size = data.data[-2:].astype(int)

    img = (np.reshape(img_raveled, (img_size[0], img_size[1], 3))).astype(np.uint8)
    image_message = CvBridge().cv2_to_imgmsg(img, encoding="rgb8")
    pub_1.publish(image_message)

def callback_2(data):
    img_raveled = data.data[0:-2]
    img_size = data.data[-2:].astype(int)

    img = (np.reshape(img_raveled, (img_size[0], img_size[1], 3))).astype(np.uint8)
    image_message = CvBridge().cv2_to_imgmsg(img, encoding="rgb8")
    pub_2.publish(image_message)

def callback_3(data):
    img_raveled = data.data[0:-2]
    img_size = data.data[-2:].astype(int)

    img = (np.reshape(img_raveled, (img_size[0], img_size[1], 3))).astype(np.uint8)
    image_message = CvBridge().cv2_to_imgmsg(img, encoding="rgb8")
    pub_3.publish(image_message)


def listener():
    
    rospy.Subscriber("rgb_1", numpy_msg(Floats), callback_1)
    rospy.Subscriber("rgb_2", numpy_msg(Floats), callback_2)
    # rospy.Subscriber("rgb_3", numpy_msg(Floats), callback_3)
    rospy.spin()


if __name__ == "__main__":
    listener()
