import os
import random
import sys

import git
import numpy as np
from gym import spaces

# %matplotlib inline
from matplotlib import pyplot as plt

# %cd "/content/habitat-lab"


from PIL import Image
import imageio

import habitat
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationTask
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps


IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


if __name__ == "__main__":
    config = get_baselines_config(
        "./habitat_baselines/config/pointnav/ddppo_pointnav.yaml"
    )

# %%
# set random seeds
if __name__ == "__main__":
    seed = "42"  # @param {type:"string"}
    # steps_in_thousands = "10"  # @param {type:"string"}

    config.defrost()
    config.TASK_CONFIG.SEED = int(seed)
    # config.TOTAL_NUM_STEPS = int(steps_in_thousands)
    config.LOG_INTERVAL = 1
    config.freeze()

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)

# %%
if __name__ == "__main__":
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    trainer = trainer_init(config)
    # actor_critic = trainer._setup_actor_critic_agent(config.RL.PPO)
    # trainer.train()
    trainer.eval()
    # agent = trainer.agent()
    # goal_radius = 0.5
    # goal = NavigationGoal(position=[10, 0.25, 10], radius=goal_radius)
    # agent_position = np.array([0, 0.25, 0])
    # agent_rotation = -np.pi / 4

    # dummy_episode = NavigationEpisode(
    #     goals=[goal],
    #     episode_id="dummy_id",
    #     scene_id="dummy_scene",
    #     start_position=agent_position,
    #     start_rotation=agent_rotation,
    # )
    # target_image = maps.pointnav_draw_target_birdseye_view(
    #     agent_position,
    #     agent_rotation,
    #     np.asarray(dummy_episode.goals[0].position),
    #     goal_radius=dummy_episode.goals[0].radius,
    #     agent_radius_px=25,
    # )

    # imageio.imsave(
    #     os.path.join(IMAGE_DIR, "pointnav_target_image.png"), target_image
    # )
