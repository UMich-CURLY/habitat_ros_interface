
import os 
import argparse
import yaml
from IPython import embed
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')
ARGS = PARSER.parse_args()
scene = ARGS.scene
dataset = ARGS.dataset



with open("configs/tasks/pointnav_rgbd.yaml",'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)
    if (dataset == "mp3d"):
        config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz"
    elif (dataset == "gibson"):
        config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/gibson/v1/test/content/"+scene+"0.json.gz"
    elif (dataset == "habitat"):
        config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/habitat-test-scenes/v1/test/content/"+scene+"0.json.gz"
with open("configs/tasks/pointnav_rgbd.yaml",'w') as file:
    print("Replacing the data config to the new scene ", scene)
    documents = yaml.dump(config, file)
with open("configs/tasks/pointnav_mp3d.yaml",'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)
    if (dataset == "mp3d"):
        config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz"
    elif (dataset == "gibson"):
        config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/gibson/v1/test/content/"+scene+"0.json.gz"
    elif (dataset == "habitat"):
        config['DATASET']['DATA_PATH'] = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/habitat-test-scenes/v1/test/content/"+scene+"0.json.gz"

with open("configs/tasks/pointnav_mp3d.yaml",'w') as file:
    print("Replacing the data config to the new scene ", scene)
    documents = yaml.dump(config, file)

__ = os.system('python ./scripts/door_rl_agent.py --scene '+ scene)