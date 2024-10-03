import os 
import path
path_1 = "/home/tribhi/research/habicrowd/habitat_ros_interface/data/datasets/irl_jun_20_7_fixed_rank"
path_2 = "/home/tribhi/research/habicrowd/habitat_ros_interface/data/datasets/irl_jun_21_2"

ranks = []

demos = os.listdir(path_1)

for demo in demos:
    print(demo)
    __ = os.system("cp "+path_1+"/"+demo + '/' + "rank.txt " + path_2+"/"+demo + '/')