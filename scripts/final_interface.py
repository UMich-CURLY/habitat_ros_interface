import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import habitat
import habitat_sim.bindings as hsim
import magnum as mn
import csv
import random
import os
import os.path as osp
import gzip
import json
# %matplotlib inline
from matplotlib import pyplot as plt
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
# function to display the topdown map
from PIL import Image


rospy.init_node("interim",anonymous=False)

def convert_points_to_topdown(pathfinder, points, meters_per_pixel = 0.5):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

env_config_file = "configs/tasks/pointnav_rgbd.yaml"

class episodal_nav():
    _current_episode = 0
    _total_number_of_episodes = 0
    observations = []

    def __init__(self):
        self.config = get_baselines_config(
        "./habitat_baselines/config/pointnav/ddppo_pointnav.yaml"
        )
         # @param {type:"string"}
        # steps_in_thousands = "10"  # @param {type:"string"}

        
        self.env = habitat.Env(config=habitat.get_config(env_config_file))
        self.observations = []
        self.observations.append(self.env.reset())
        
        # self.env._sim.agents[0].state.velocity = np.float32([0, 0, 0])
        # self.env._sim.agents[0].state.angular_velocity = np.float32([0, 0, 0])
        # self.env._sim.agents[0].state.position = np.float32([-5.793274826959732,0.07,-1.4784819169477985])
        # self.env._sim.agents[0].state.rotation = np.float32([0.0,0.0,0.0,1.0])
        self.env._sim.set_agent_state(np.float32([-5.793274826959732,0.07,-1.4784819169477985]), np.float32([0.0,0.0,0.0,1.0]))
    def create_episode(self):
        sim = self.env._sim

        dset = habitat.datasets.make_dataset("PointNav-v1")
              
        goal_radius = 0.05
        goal_position = []
        goal = NavigationGoal(position=[-0.2931181443940609,0.07,0.4782877141779114])
        state = sim.get_agent_state()
        print(state)
        agent_position = state.position
        agent_rotation = state.rotation

        dummy_episode = NavigationEpisode(
        goals=[goal],
        episode_id=0,
        scene_id="data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
        start_position=agent_position,
        start_rotation=agent_rotation,
        )
        dset.episodes = [dummy_episode]
        # for ep in dset.episodes:
        #     ep.scene_id = "data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
        print(dset.episodes)
        # scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
        self.out_file = f"./data/datasets/pointnav/mp3d/v1/test/content/17DRP5sb8fy" + str(self._current_episode)+".json.gz"
        os.makedirs(osp.dirname(self.out_file), exist_ok=True)
        with gzip.open(self.out_file, "wt") as f:
            f.write(dset.to_json())

    def run_episode(self):
        seed = "42"      
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = int(seed)
        self.config.LOG_INTERVAL = 1
        self.config.TASK_CONFIG.DATASET.DATA_PATH = self.out_file
        self.config.freeze()
        print(self.config.TASK_CONFIG.DATASET)
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)  
        trainer_init = baseline_registry.get_trainer(self.config.TRAINER_NAME)
        trainer = trainer_init(self.config)
        policy = trainer.eval()
        while not rospy.is_shutdown():
            print("rospy running")
        rospy.init_node("fuck",anonymous=False)
        self._pub_policy = rospy.Publisher("~policy", numpy_msg(Floats), queue_size=1)
        print(trainer._final_policy_hack_tribhi)
        for actions in trainer._final_policy_hack_tribhi:
            self.observations.append(self.env.step(actions[0]))
        self._pub_policy.publish(np.float32(trainer._final_policy_hack_tribhi))
        # make_video = True
        # if make_video:
        #     overlay_dims = (int(256/ 5), int(256 / 5))
        #     print("overlay_dims = " + str(overlay_dims))
        #     overlay_settings = [
        #         {
        #             "obs": "rgb",
        #             "type": "color",
        #             "dims": overlay_dims,
        #             "pos": (10, 10),
        #             "border": 2,
        #         },
        #         {
        #             "obs": "depth",
        #             "type": "depth",
        #             "dims": overlay_dims,
        #             "pos": (10, 30 + overlay_dims[1]),
        #             "border": 2,
        #         },
        #     ]
        #     print("overlay_settings = " + str(overlay_settings))

        #     vut.make_video(
        #         observations=self.observations,
        #         primary_obs="rgb",
        #         primary_obs_type="color",
        #         video_file="video_dir/translate",
        #         fps=int(1.0 / 60),
        #         open_vid= True,
        #         overlay_settings=overlay_settings,
        #         depth_clip=10.0,
        #     )



def main():
    navigate = episodal_nav()
    navigate.create_episode()
    
    navigate.run_episode()
    while not rospy.is_shutdown():
        print("rospy running")

if __name__ == "__main__":
    main()