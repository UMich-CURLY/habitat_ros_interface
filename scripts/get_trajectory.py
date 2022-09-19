from pathlib import Path
import numpy as np
import sys
sys.path.append("/Py_Social_ROS")
import pysocialforce as psf
from PIL import Image
import numpy as np
import yaml
import itertools


def get_velocity(initial_state, groups = None):
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of (px, py, vx, vy, gx, gy)
    multiplier = 1
    # initial_state = np.array(
    #     [
    #         [200, 150,1.0*multiplier, 0.0*multiplier, 300, 150],
    #         [200, 155,1.0*multiplier, 0.0*multiplier, 300, 155]
    #         # [300, 100, 0.0*multiplier, -1.0*multiplier, 400, 150],
    #         # [400, 150, -2.0*multiplier, 0.0*multiplier, 300, 100],
    #         # [100, 30, 0.0*multiplier, 2.0*multiplier, 50, 100],
    #         # [1.0, 1.0, 0.0, 0.5, -2.0, 1.0],
    #         # [2.0, 0.0, 0.0, 0.5, 3.0, 10.0],
    #         # [0.0, 6.0, 0.5, 0.5, 5.0, 6.0],
    #         # [5.0, 6.0, -0.5, 0.5, 0.0, 6.0],
    #     ]
    # )
    # # social groups informoation is represented as lists of indices of the state array
    # groups = [[0],[1]]

    obs = []
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    # for i in np.arange(0,10,0.1):
    #     obs.append([-1,i])
    # for i in np.arange(0,10,0.1):
    #     obs.append([3+i,i])
    #obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    img = Image.open("default.pgm").convert('L')
    # img.show()
    img_np = np.array(img)  # ndarray
    white=0
    wall=0
    space=0
    # obs = []
    for i in np.arange(img_np.shape[0]):
        for j in np.arange(img_np.shape[1]):
            if img_np[i][j]== 255:  # my-map 254 ->space, 0 -> wall, 205-> nonspace
                white=white+1
                # obs.append([j,i])
            if img_np[i][j]== 0:    # sample-map 128 -> space, 0 -> wall, 255-> nonspace
                wall=wall+1
                obs.append([j,i])
            if img_np[i][j]== 128:
                space=space+1 
    print(i,j)
    # for i in np.arange(img_np.shape[0]):
    #     for j in np.arange(img_np.shape[1]):
    #         if img_np[i][j] != 255 and img_np[i][j] != 0:
    #            print(img_np[i][j])
    # print(white)
    # print(space)
    # print(wall)
    # print(img_np[160:170,320:330])
    # print(img_np.shape)
    #obs = [[1, 2, 7, 8]]
    # obs = img_np[0:4,0:10]
    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        groups=groups,
        obstacles=obs,
        config_file=Path(__file__).resolve().parent.joinpath("examples/example.toml"),
    )
    # update 80 steps
    s.step(1)
    print("States" ,s.peds.get_states())
    return s.vel()
    # with psf.plot.SceneVisualizer(s, "images/result_2") as sv:
    #     sv.animate()
        # sv.plot()
