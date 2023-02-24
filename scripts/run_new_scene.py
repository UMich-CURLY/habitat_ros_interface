import os 
import argparse
from IPython import embed
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
ARGS = PARSER.parse_args()
scene = ARGS.scene

if (not os.path.isfile("./data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz")):     
    __ = os.system('python ./scripts/rearrangement_episode.py --scene '+ scene)
__ = os.system('python ./scripts/get_topdown_map.py --scene '+ scene + ' --mps 0.025')
__ = os.system('python ./scripts/follower_and_robot.py --scene '+ scene)
