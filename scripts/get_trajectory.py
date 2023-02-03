from pathlib import Path
import numpy as np
import sys
sys.path.append("/Py_Social_ROS")
sys.path.append("/PySocialForce")
import pysocialforce as psf
from PIL import Image
import numpy as np
import yaml
import itertools
from IPython import embed


class social_force():
    
    obs = []
    # def __init__(self):
        # img = Image.open("/Py_Social_ROS/default.pgm").convert('L')
        # img.show()
        # img_np = np.array(img)  # ndarray
        # white=0
        # wall=0
        # space=0
        # obs = []
        # for i in np.arange(img_np.shape[0]):
        #     for j in np.arange(img_np.shape[1]):
        #         if img_np[i][j]== 255:  # my-map 254 ->space, 0 -> wall, 205-> nonspace
        #             white=white+1
        #             # obs.append([j,i])
        #         if img_np[i][j]== 0:    # sample-map 128 -> space, 0 -> wall, 255-> nonspace
        #             wall=wall+1
        #             self.obs.append([j,i])
        #         if img_np[i][j]== 128:
        #             space=space+1 
    def load_obstacles(self, env):
        # Add scenes objects to ORCA simulator as obstacles
        sem_scene = env._sim.semantic_annotations()

        for obj in sem_scene.objects:
            if obj.category.name() in ['floor', 'ceiling']:
                continue
            center = obj.aabb.center
            x_len, _, z_len = (
                obj.aabb.sizes / 2.0
            )
            # Nodes to draw rectangle
            corners = [
                center + np.array([x, 0, z])
                for x, z in [
                    (-x_len, -z_len),
                    (-x_len, z_len),
                    (x_len, z_len),
                    (x_len, -z_len),
                    (-x_len, -z_len),
                ]
            ]

            # map_corners = [
            #     maps.to_grid(
            #         p[2],
            #         p[0],
            #         (
            #             self.grid_dimensions[0],
            #             self.grid_dimensions[1],
            #         ),
            #         sim=self.env._sim,
            #     )
            #     for p in corners
            # ]
            self.obs.append([corners[0][0], corners[0][2], corners[1][0], corners[1][2]])
            self.obs.append([corners[1][0], corners[1][2], corners[2][0], corners[2][2]])
            self.obs.append([corners[2][0], corners[2][2], corners[3][0], corners[3][2]])
            self.obs.append([corners[3][0], corners[3][2], corners[4][0], corners[4][2]])
    def get_velocity(self,initial_state, current_heading = None, groups = None, filename = None, save_anim = False):
        # initiate the simulator,
        
        s = psf.Simulator(
            initial_state,
            groups=groups,
            obstacles=self.obs,
            config_file=Path(__file__).resolve().parent.joinpath("/PySocialForce/examples/example.toml"),
        )
        # update 80 steps
        
        if(save_anim):
            s.step(50)
            with psf.plot.SceneVisualizer(s, "/PySocialForce/images/"+filename) as sv:
                # sv.animate()
                sv.plot()
        s.step(1)
        # print("Agent radius is", s.peds.agent_radius)
        return s.peds.vel()
    
    
