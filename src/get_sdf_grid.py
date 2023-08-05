#!/usr/bin/env python
# note need to run viewer with python2!!!

import rospy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from IPython import embed
import argparse
import sys
import os
sys.path.append(os.path.abspath('/home/catkin_ws/src/habitat_ros_interface'))
from srv import sdf_grid
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
ARGS = PARSER.parse_args()
scene = ARGS.scene
dist_map_file = "/home/catkin_ws/src/habitat_ros_interface/maps/sdf_resolution_"+scene+"_0.025.pgm"
img = cv.imread(dist_map_file)

def get_grid(req):
    [grid_x, grid_y] = [req.x*0.025, req.y*0.025]
    return img[grid_x,grid_y]

if __name__ == "__main__":
    rospy.init_node('get_sdf_grid_server')   
    s = rospy.Service('get_sdf_grid', sdf_grid, get_grid)
    rospy.spin()
