from pathlib import Path
import numpy as np
import sys
import rvo2
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
import toml
import matplotlib.pyplot as plt

def my_ceil(a, precision=2):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=2):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)

class ped_rvo():
    
    obs = []
    def __init__(self, my_env, map_path, config_file = "./scripts/rvo2_default.toml", resolution = 0.025):
        num_sqrt_meter = np.sqrt(my_env.grid_dimensions[0] * my_env.grid_dimensions[1]*0.025*0.025)
        self.config = {}
        user_config = toml.load(config_file)
        self.config.update(user_config)
        self.num_sqrt_meter_per_ped = self.config.get(
            'num_sqrt_meter_per_ped', 8)
        self.num_pedestrians = max(1, int(
            num_sqrt_meter / self.num_sqrt_meter_per_ped))

        """
        Parameters for our mechanism of preventing pedestrians to back up.
        Instead, stop them and then re-sample their goals.

        num_steps_stop         A list of number of consecutive timesteps
                               each pedestrian had to stop for.
        num_steps_stop_thresh  The maximum number of consecutive timesteps
                               the pedestrian should stop for before sampling
                               a new waypoint.
        neighbor_stop_radius   Maximum distance to be considered a nearby
                               a new waypoint.
        backoff_radian_thresh  If the angle (in radian) between the pedestrian's
                               orientation and the next direction of the next
                               goal is greater than the backoffRadianThresh,
                               then the pedestrian is considered backing off.
        """
        self.num_steps_stop = [0] * self.num_pedestrians
        self.neighbor_stop_radius = self.config.get(
            'neighbor_stop_radius', 1.0)
        # By default, stop 2 seconds if stuck
        self.num_steps_stop_thresh = self.config.get(
            'num_steps_stop_thresh', 20)
        # backoff when angle is greater than 135 degrees
        self.backoff_radian_thresh = self.config.get(
            'backoff_radian_thresh', np.deg2rad(135.0))

        """
        Parameters for ORCA

        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        """
        self.neighbor_dist = self.config.get('orca_neighbor_dist', 2)
        self.max_neighbors = self.num_pedestrians
        self.time_horizon = self.config.get('orca_time_horizon', 1.0)
        self.time_horizon_obst = self.config.get('orca_time_horizon_obst', 0.5)
        self.orca_radius = self.config.get('orca_radius', 0.2)
        self.orca_max_speed = self.config.get('orca_max_speed', 0.5)
        self.dt = my_env.human_time_step
        self.orca_sim = rvo2.PyRVOSimulator(
            my_env.human_time_step,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.time_horizon_obst,
            self.orca_radius,
            self.orca_max_speed)
        print("About to load obs")
        resolution = 0.01
        self.load_obs_from_map(map_path, resolution)
        self.fig, self.ax = plt.subplots()
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
    
    def load_obs_from_map(self, map_path, resolution):
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
                    self.orca_sim.addObstacle([tuple([my_floor(j*resolution), my_floor(i*resolution)]), tuple([my_ceil(j*resolution),my_floor(i*resolution)]), tuple([my_ceil(j*resolution),my_ceil(i*resolution)]), tuple([my_floor(j*resolution), my_ceil(i*resolution)])])
                    # self.orca_sim.addObstacle([tuple([my_floor(j/40), my_floor(i/40)]), tuple([my_floor(j/40), my_floor(i/40)]), tuple([my_floor(j/40), my_floor(i/40)]), tuple([my_floor(j/40), my_floor(i/40)])])
                    # self.orca_sim.addObstacle([tuple([j/40, i/40]), tuple([j/40,i/40]), tuple([j/40,i/40]), tuple([j/40, i/40])])
                if img_np[i][j]== 128:
                    space=space+1
        print("building the obstacle tree")
        self.orca_sim.processObstacles()

    def get_velocity(self,initial_state, current_heading = None, groups = None, filename = None, save_anim = False):
        self.orca_ped = []
        for i in range(len(initial_state)):
            self.orca_ped.append(self.orca_sim.addAgent((initial_state[i][0],initial_state[i][1]), velocity = (initial_state[i][2], initial_state[i][3])))
            desired_vel = np.array([initial_state[i][4] - initial_state[i][0], initial_state[i][5]-initial_state[i][1]]) 
            desired_vel = desired_vel/np.linalg.norm(desired_vel) * self.orca_max_speed
            self.orca_sim.setAgentPrefVelocity(self.orca_ped[-1], tuple(desired_vel))
        self.orca_sim.doStep()
        computed_velocity=[]
        for j in range(len(initial_state)):
            [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
            velx = (x - initial_state[j][0])/self.dt
            vely = (y - initial_state[j][1])/self.dt
            computed_velocity.append([velx,vely])
        print(computed_velocity)
        if save_anim:
            self.plot_obstacles()
            num_steps = 1000
            for i in range(num_steps):
                self.orca_sim.doStep()
                colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
                vel = []
                for j in range(len(initial_state)):
                    [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
                    velx = (x - initial_state[j][0])/self.dt
                    vely = (y - initial_state[j][1])/self.dt
                    initial_state[j][0] = x
                    initial_state[j][1] = y
                    vel.append([velx,vely])
                    print(velx,vely)
                    self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=0.5, color=colors[j])
                if np.all(vel[0] == [0.0,0.0] and vel[1] == [0.0,0.0]):
                    break
            self.fig.savefig(filename+".png", dpi=300)
            plt.close(self.fig)
        return np.array(computed_velocity)
            
    def get_position(self,initial_state, current_heading = None, groups = None, filename = None, save_anim = False):
        self.orca_ped = []
        for i in range(len(initial_state)):
            self.orca_ped.append(self.orca_sim.addAgent((initial_state[i][0],initial_state[i][1]), velocity = (initial_state[i][2], initial_state[i][3])))
            desired_vel = np.array([initial_state[i][4] - initial_state[i][0], initial_state[i][5]-initial_state[i][1]]) 
            desired_vel = desired_vel/np.linalg.norm(desired_vel) * self.orca_max_speed
            self.orca_sim.setAgentPrefVelocity(self.orca_ped[-1], tuple(desired_vel))
        self.orca_sim.doStep()
        computed_velocity=[]
        for j in range(len(initial_state)):
            [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
            velx = (x - initial_state[j][0])/self.dt
            vely = (y - initial_state[j][1])/self.dt
            computed_velocity.append([velx,vely])
        print(computed_velocity)
        if save_anim:
            self.plot_obstacles()
            num_steps = 1000
            for i in range(num_steps):
                self.orca_sim.doStep()
                colors = plt.cm.rainbow(np.linspace(0, 1, len(initial_state)))
                vel = []
                for j in range(len(initial_state)):
                    [x,y] = self.orca_sim.getAgentPosition(self.orca_ped[j])
                    velx = (x - initial_state[j][0])/self.dt
                    vely = (y - initial_state[j][1])/self.dt
                    initial_state[j][0] = x
                    initial_state[j][1] = y
                    vel.append([velx,vely])
                    print(velx,vely)
                    self.ax.plot(x, y, "-o", label=f"ped {j}", markersize=2.5, color=colors[j])
                if np.all(vel[0] == [0.0,0.0] and vel[1] == [0.0,0.0]):
                    break
            self.fig.savefig(filename+".png", dpi=300)
            plt.close(self.fig)
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
        obstacles = []
        for i in range(self.orca_sim.getNumObstacleVertices()):
            obstacles.append(self.orca_sim.getObstacleVertex(i))
        xy_limits=np.array(obstacles)
        xmin = 10000
        ymin = 10000
        xmax = -10000
        ymax = -10000
        for obs in xy_limits:
            xmin = min(xmin,obs[0])
            xmax = max(xmax,obs[0])
            ymin = min(ymin,obs[1])
            ymax = max(ymax,obs[1])
        self.ax.set(xlim=(xmin-2,xmax+3), ylim=(ymin-2, ymax+3))
        self.ax.plot(xy_limits[:, 0], xy_limits[:, 1], "o", color="black", markersize=0.1)
    
