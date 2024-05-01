#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
import os
rospy.init_node("restart_map",anonymous=False)


def callback_4(msg):
    if msg.data:
        print("Relaunching map server")
        __ = os.system("rosnode kill map_server")
        # __ = os.system("rosrun map_server map_server /home/catkin_ws/src/habitat_ros_interface/maps/sample_map.yaml")
        return True

def listener():
    
    rospy.Subscriber("/reload_map_server", Bool, callback_4)
    rospy.spin()


if __name__ == "__main__":
    listener()