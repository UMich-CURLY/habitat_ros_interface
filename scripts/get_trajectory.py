from pathlib import Path
import numpy as np
import sys
sys.path.append("/Py_Social_ROS")
import pysocialforce as psf
from PIL import Image
import numpy as np
import yaml
import itertools

class social_force():
    
    obs = []
    def __init__(self):
        img = Image.open("/Py_Social_ROS/default.pgm").convert('L')
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
                    self.obs.append([j,i])
                if img_np[i][j]== 128:
                    space=space+1 
    def get_velocity(self,initial_state, current_heading = None, groups = None, filename = None):
        # initiate the simulator,
        s = psf.Simulator(
            initial_state,
            groups=groups,
            obstacles=self.obs,
            config_file=Path(__file__).resolve().parent.joinpath("/Py_Social_ROS/examples/example.toml"),
        )
        # update 80 steps
        s.step(1)
        
        # with psf.plot.SceneVisualizer(s, "/Py_Social_ROS/images/"+filename) as sv:
        # #     sv.animate()
        #     sv.plot()
        return s.peds.vel()
