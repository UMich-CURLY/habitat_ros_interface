import os 
import argparse
from IPython import embed
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')
ARGS = PARSER.parse_args()
scene = ARGS.scene
dataset = ARGS.dataset
if dataset =="gibson":
    if (not os.path.isfile("./data/datasets/pointnav/gibson/v1/test/content/"+scene+"0.json.gz")):     
        __ = os.system('python ./scripts/creating_episodes.py --scene '+ scene +' --dataset ' + dataset)
    if (not os.path.isfile("./data/datasets/rearrange/gibson/v1/test/content/"+scene+"0.json.gz")):     
        __ = os.system('python ./scripts/rearrangement_episode.py --scene '+ scene +' --dataset ' + dataset)
    
elif dataset == "hm3d":
    if (not os.path.isfile("./data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz")):     
        __ = os.system('python ./scripts/rearrangement_episode_hm3d.py --scene '+ scene)
else:
    # if (not os.path.isfile("./data/datasets/pointnav/mp3d/v1/test/content/"+scene+"0.json.gz")):     
    __ = os.system('python ./scripts/_episodes.py --scene '+ scene+' --dataset ' + dataset)
    if (not os.path.isfile("./data/datasets/rearrange/mp3d/v1/test/content/"+scene+"0.json.gz")):     
        __ = os.system('python ./scripts/rearrangement_episode.py --scene '+ scene+' --dataset ' + dataset)
    
__ = os.system('python ./scripts/get_topdown_map_old.py --scene '+ scene + ' --mps 0.025 --dataset ' + dataset)
__ = os.system('python ./maps/get_outline.py --scene '+ scene)
__ = os.system('python ./scripts/get_sdf.py --scene '+ scene)

# __ = os.system('python ./scripts/get_topdown_map_old.py --scene '+ scene + ' --mps 0.025')
__ = os.system('python ./scripts/get_topdown_map.py --scene '+ scene + ' --mps 0.025 --dataset ' + dataset)
# __ = os.system('python ./scripts/follower_and_robot.py --scene '+ scene)
__ = os.system('python ./scripts/door_refactor.py --scene '+ scene)
