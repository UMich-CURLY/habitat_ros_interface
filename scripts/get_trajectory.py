from pathlib import Path
import numpy as np
import sys
sys.path.append("/Py_Social_ROS")
# sys.path.append("/PySocialForce")
import pysocialforce as psf
from PIL import Image
import numpy as np
import yaml
import itertools
from IPython import embed
import tf
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt

def my_ceil(a, precision=2):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=2):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

class social_force():
    
    obs = []
    
    def __init__(self,my_env, map_path, resolution = 0.025, groups = [[0,1]]):
        self.load_obs_from_map(map_path)
        groups = my_env.groups
        self.dt = my_env.human_time_step
        print("About to load obs")
        self.load_obs_from_map(map_path, resolution)
        self.s = psf.Simulator(
            np.array(my_env.initial_state),
            groups=groups,
            obstacles=self.obs,
            config_file=Path(__file__).resolve().parent.joinpath("/Py_Social_ROS/examples/example.toml"),
        )
        self.fig, self.ax = plt.subplots()
        self.save_plot = False
        if (self.save_plot):
            self.plot_obstacles()
        self.max_counter = int(20/my_env.human_time_step)
        self.update_number = 0
        self.dt = my_env.human_time_step
        self.s.peds.step_width = 0.5*my_env.human_time_step
        
        
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
            quat = tf.transformations.quaternion_inverse(obj.obb.rotation)
            trans = tf.transformations.quaternion_matrix(quat)
            map_corners = [
                np.dot(trans,np.array([p[0],p[1],p[2],1.0]))
                for p in corners
            ]
            map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                40,
                                74,
                            ),
                            sim=env._sim,
                        )
                        for p in corners
                    ]
            # map_corners = [
            #     np.dot(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),np.array([p[0],p[1],p[2],1.0]))
            #     for p in corners
            # ]
            self.obs.append([map_corners[0][0], map_corners[1][0], map_corners[0][1], map_corners[1][1]])
            self.obs.append([map_corners[1][0], map_corners[2][0], map_corners[1][1], map_corners[2][1]])
            self.obs.append([map_corners[2][0], map_corners[3][0], map_corners[2][1], map_corners[3][1]])
            self.obs.append([map_corners[3][0], map_corners[4][0], map_corners[3][1], map_corners[4][1]])
    
    def load_obs_from_map(self, map_path, resolution = 0.025):
        img = Image.open(map_path).convert('L')
        # img.show() 
        img_np = np.array(img)  # ndarray
        white=0
        wall=0
        space=0
        for i in np.arange(img_np.shape[0]):
            for j in np.arange(img_np.shape[1]):
                if img_np[i][j]== 255:  # my-map 254 ->space, 0 -> wall, 205-> nonspace
                    white=white+1
                    # obs.append([j,i])
                if img_np[i][j]== 0:    # sample-map 128 -> space, 0 -> wall, 255-> nonspace
                    wall=wall+1
                    self.obs.append([my_floor(j*resolution), my_ceil(j*resolution),my_floor(i*resolution),my_ceil(i*resolution)])
                if img_np[i][j]== 128:
                    space=space+1 

    def get_velocity(self,initial_state, current_heading = None, groups = None, filename = None, save_anim = False):
        # initiate the simulator,
        # update 80 steps
        for i in range(len(initial_state)):
            self.s.peds.state[i,0:6] = initial_state[i][0:6]
        self.s.step(1)
        if (self.save_plot):
            colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
            alpha = np.linspace(0.5,1,self.max_counter+1)
        computed_velocity=[]
        for j in range(len(initial_state)):
            [x,y] = self.s.peds.state[j,0:2]
            velx = (x - initial_state[j][0])/self.dt
            vely = (y - initial_state[j][1])/self.dt
            computed_velocity.append([velx,vely])
            if (self.update_number < self.max_counter and self.save_plot):
                self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=2.5, color=colors[j], alpha = alpha[self.update_number])
                self.ax.plot(initial_state[j][4], initial_state[j][5], "-x", label=f"ped {j}", markersize=2.5, color=colors[j], alpha = alpha[self.update_number])
        if (self.update_number == self.max_counter and self.save_plot):
            print("saving the offline plot!!")
            self.fig.savefig("save_stepwise_esfm"+".png", dpi=300)
            plt.close(self.fig)
        self.update_number+=1
        ### Find out how to update agent positions in this ####
        print("Velocity returned is ", computed_velocity)
        print("State of the agent is ", self.s.peds.state)
        return np.array(computed_velocity)
    
    def plot_obstacles(self):
        self.fig.set_tight_layout(True)
        self.ax.grid(linestyle="dotted")
        self.ax.set_aspect("equal")
        self.ax.margins(2.0)
        self.ax.set_axisbelow(True)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        plt.rcParams["animation.html"] = "jshtml"

        # x, y limit from states, only for animation
        margin = 2.0 
        xy_limits=np.array(self.obs)
        xmin = 10000
        ymin = 10000
        xmax = -10000
        ymax = -10000
        for obs in xy_limits:
            xmin = min(xmin,obs[0])
            xmax = max(xmax,obs[1])
            ymin = min(ymin,obs[2])
            ymax = max(ymax,obs[3])
        self.ax.set(xlim=(xmin-2,xmax+3), ylim=(ymin-2, ymax+3))
        for k in self.s.get_obstacles():
            self.ax.plot(k[:, 0], k[:, 1], "-o", color="black", markersize=0.1)
