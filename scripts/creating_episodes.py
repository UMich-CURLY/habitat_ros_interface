#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import glob
import gzip
import json
import multiprocessing
import os
import os.path as osp

import tqdm

import habitat
import habitat_sim
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.datasets.rearrange.rearrange_dataset import *
num_episodes_per_scene = 1

def _create_episode(art_objs: List[List[Any]],
    static_objs: List[List[Any]],
    targets: List[List[Any]],
    fixed_base: bool,
    art_states: List[Any],
    nav_mesh_path: str,
    scene_config_path: str,
    allowed_region: List[Any] = [],
    markers: List[Dict[str, Any]] = [],
    force_spawn_pos: List = None
) -> Optional[RearrangeEpisode]:
    
    return RearrangeEpisode(
        episode_id=str(episode_id),
        goals=goals,
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        shortest_paths=shortest_paths,
        info=info,
    )

def generate_rearrangement_episode():
    episode_count = 0
    num_episodes = 1
    while episode_count < num_episodes or num_episodes < 0:
        target_position = sim.sample_navigable_point()

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue

        for _retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )
            if is_compatible:
                break
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            shortest_paths = None
            if is_gen_shortest_path:
                try:
                    shortest_paths = [
                        get_action_shortest_path(
                            sim,
                            source_position=source_position,
                            source_rotation=source_rotation,
                            goal_position=target_position,
                            success_distance=shortest_path_success_distance,
                            max_episode_steps=shortest_path_max_steps,
                        )
                    ]
                # Throws an error when it can't find a path
                except GreedyFollowerError:
                    continue

            episode = _create_episode(
                episode_id=episode_count,
                scene_id=sim.habitat_config.SCENE,
                start_position=source_position,
                start_rotation=source_rotation,
                target_position=target_position,
                shortest_paths=shortest_paths,
                radius=shortest_path_success_distance,
                info={"geodesic_distance": dist},
            )

            episode_count += 1
            yield episode

def _generate_fn(scene):
    cfg = habitat.get_config("/habitat-lab/configs/tasks/rearrangepick_replica_cad.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = scene
    cfg.SIMULATOR.AGENT_0.SENSORS = []
    cfg.freeze()

    sim = habitat.sims.make_sim("RearrangeSim-v0", config=cfg.SIMULATOR)
    sim.reconfigure(cfg)
    print(sim.ep_info)
    # pointnav_data = PointNavDatasetV1("configs/tasks/try_rearrange.yaml")
    # rearrangement_dataset = RearrangeDatasetV0(sim)
    # print(rearrangement_dataset.episodes)
    # rearrange_dataset.episodes = list(generate_rearrangement_episode(sim, num_episodes_per_scene))

    # dset = habitat.datasets.make_dataset("RearrangeDataset-v0")
    # dset.episodes = list(
    #     generate_pointnav_episode(
    #         sim, num_episodes_per_scene, is_gen_shortest_path=False
    #     )
    # )
    # count_episodes = 0
    
    # for ep in rearrangement_dataset.episodes:
    #     ep.scene_id = "/habitat-lab/scene_datasets/mp3d/Vt2qJdWjCF2/Vt2qJdWjCF2.glb"
    # print(rearrangement_dataset.episodes)
    # scene_key = osp.basename(osp.dirname(osp.dirname(scene)))
    # out_file = f"./data/datasets/pointnav/mp3d/v1/test/content/Vt2qJdWjCF2" + str(count_episodes)+".json.gz"
    # os.makedirs(osp.dirname(out_file), exist_ok=True)
    # with gzip.open(out_file, "wt") as f:
    #     f.write(rearrangement_dataset.to_json())



# scenes = glob.glob("./data/scene_datasets/mp3d/17DRP5sb8fy.glb")
# with multiprocessing.Pool(8) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
#     for _ in pool.imap_unordered(_generate_fn, scenes):
#         pbar.update()

# with gzip.open(f"./data/datasets/pointnav/mp3d/v1/all/all.json.gz", "wt") as f:
#     json.dump(dict(episodes=[]), f)

if __name__ == "__main__":
	_generate_fn("/habitat-lab/scene_datasets/mp3d/Vt2qJdWjCF2/Vt2qJdWjCF2.glb")