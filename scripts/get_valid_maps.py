import os 
import argparse
from IPython import embed
PARSER = argparse.ArgumentParser(description=None)

PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')
ARGS = PARSER.parse_args()
dataset = ARGS.dataset

DATA_PATH = "./data/scene_datasets/mp3d"
invalid_scenes = []
for foldername in os.listdir(DATA_PATH):
    scene = foldername

    x = os.system('python ./scripts/creating_pointnav_dataset.py --scene '+ scene+' --dataset ' + dataset)

    if (x == 0):
        __ = os.system('python ./scripts/get_topdown_map_rl.py --scene '+ scene )
        __ = os.system('python ./maps/get_outline.py --scene '+ scene)
        __ = os.system('python ./scripts/get_sdf.py --scene '+ scene)
    else:
        invalid_scenes.append(foldername)
print("These scenes did not have a solution ", invalid_scenes)
# __ = os.system('python ./scripts/get_topdown_map_old.py --scene '+ scene + ' --mps 0.025')
# __ = os.system('python ./scripts/get_topdown_map.py --scene '+ scene + ' --mps 0.025 --dataset ' + dataset)
# __ = os.system('python ./scripts/follower_and_robot.py --scene '+ scene)
# __ = os.system('python ./scripts/door_rl_agent.py --scene '+ scene)
