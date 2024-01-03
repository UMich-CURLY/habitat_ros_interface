
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

__ = os.system('python ./scripts/get_topdown_map.py --scene '+ scene + ' --mps 0.025 --dataset ' + dataset)
__ = os.system('python ./scripts/rearrangement_episode.py --scene '+ scene +' --dataset ' + dataset)

os.system('python ./scripts/door_driving_agent.py --scene '+ scene)
    
    
