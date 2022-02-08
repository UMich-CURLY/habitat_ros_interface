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
import threading
import tf
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
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.tasks.utils import cartesian_to_polar
# function to display the topdown map
from PIL import Image

lock = threading.Lock()
env_config_file = "configs/tasks/pointnav_rgbd.yaml"

def convert_points_from_topdown(point_2d, meters_per_pixel = 0.5):
    lower_bound = np.array([-11.59344,-0.127553,-5.392021])
    upper_bound = np.array([4.757026,4.278783,2.88662])
    size_x = upper_bound[0]-lower_bound[0]
    size_y = upper_bound[2]-lower_bound[2]
    point = [0,0,0]
    point[0] = point_2d[0]*meters_per_pixel + lower_bound[0]
    point[1] = 0.07
    point[2] = point_2d[1]*meters_per_pixel + lower_bound[2]
    return point

def check_transform():
    point_2d = [8.564516067504883, 5.1479034423828125, 0.0]
    print(convert_points_from_topdown(point_2d))


class episodal_nav():
    rospy.init_node("habitat", anonymous=False)
    _current_episode = 0
    _total_number_of_episodes = 0
    _nodes = []
    _global_plan_published = False
    _x_axis = 0
    _y_axis = 1
    _z_axis = 2
    _dt = 0.00478
    _sensor_rate = 50  # hz
    _r = rospy.Rate(_sensor_rate)
    current_goal = []
    current_position = []
    current_orientation = []

    def plan_callback(self,msg):
        if(self._global_plan_published == False):
            self._global_plan_published = True
            length = len(msg.data)
            self._nodes = msg.data.reshape(int(length/3),3)
            self._total_number_of_episodes = self._nodes.shape[0]
            self.current_position = self._nodes[self._current_episode]
            self.current_orientation = [0,0,0,1]
            self.current_goal = self._nodes[self._current_episode+1]
            self._current_episode+=1
            self.create_episode()
            self.run_episode()
            print("Exiting plan_callback")

    def callback(self, msg):
        self.policy = list(map(int,msg.data))
        self.policy.reverse()
        print(self.policy)
        print("Generating and running next episode")
        self.current_goal = self._nodes[self._current_episode+1]
        self._current_episode +=1
        self.create_episode()
        self.run_episode()

# Fix the pose, convert back into 3D point!!!

    def pose_callback(self, msg):
        self.current_position = convert_points_from_topdown([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
        self.current_position = self.pathfinder.snap_point(self.current_position)
        quat = tf.transformations.quaternion_from_euler(0.0,self.heading+np.pi,0.0)
        self.current_orientation = [quat[0],quat[1],quat[2],quat[3]]
        self.env._sim.set_agent_state(np.float32(self.current_position), np.float32(self.current_orientation))
        # sim_obs = self.env.sim.get_sensor_observations()
        # obs = self.env._sim._sensor_suite.get_observations(sim_obs)
        # print(obs)
        # rgb_with_res = np.concatenate(
        #     (
        #         np.float32(obs["rgb"].ravel()),
        #         np.array(
        #             [256,256]
        #         ),
        #     )
        # )
        # self._pub_rgb.publish(np.float32(rgb_with_res))

        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses.append(msg)
        self._pub_path_msg.publish(self.path_msg)
        self._r.sleep()

    def heading_callback(self,msg):
        self.heading = msg.data[0]
        print(self.heading)

    def __init__(self):
        threading.Thread.__init__(self)
        self.config = get_baselines_config(
        "./habitat_baselines/config/pointnav/ddppo_pointnav.yaml"
        )
        env_config_file = "configs/tasks/pointnav_rgbd.yaml"        
        self.env = habitat.Env(config=habitat.get_config(env_config_file))
        self.pathfinder = self.env._sim.pathfinder
        rospy.Subscriber("~policy", numpy_msg(Floats), self.callback,  queue_size=1)
        rospy.Subscriber("~heading", numpy_msg(Floats), self.heading_callback,  queue_size=1)
        rospy.Subscriber("~pose", PoseStamped, self.pose_callback,  queue_size=1)
        rospy.Subscriber("get_points/plan_3d", numpy_msg(Floats), self.plan_callback,  queue_size=1)
        self.policy = []
        self.path_msg = Path()
        self.path_msg.header.frame_id = "world"
        self._pub_path_msg = rospy.Publisher("~actual_path", Path, queue_size=1)
        # additional RGB sensor I configured

        print("created habitat_plant succsefully")

    
    def create_episode(self):

        dset = habitat.datasets.make_dataset("PointNav-v1")
              
        goal_radius = 0.05
        goal_position = []
        goal = NavigationGoal(position=self.current_goal)
        agent_position = np.float32(self.current_position)
        agent_rotation = np.float32(self.current_orientation)
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
        print("Succesfully created episode")

    def run_episode(self):
        seed = "142"      
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = int(seed)
        self.config.LOG_INTERVAL = 1
        self.config.TASK_CONFIG.DATASET.DATA_PATH = self.out_file
        self.config.freeze()
        print(self.config.TASK_CONFIG.DATASET)
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)  
        trainer_init = baseline_registry.get_trainer(self.config.TRAINER_NAME)
        print(self.config.TRAINER_NAME)
        trainer = trainer_init(self.config)
        trainer.eval()


def main():
    
    navigate = episodal_nav()

    while ((navigate._current_episode<navigate._total_number_of_episodes) or navigate._global_plan_published == False) and not rospy.is_shutdown():
        print(navigate._current_episode)
        print(navigate._total_number_of_episodes)
        rospy.spin()

    
    # print("back in main")



if __name__ == "__main__":
    main()