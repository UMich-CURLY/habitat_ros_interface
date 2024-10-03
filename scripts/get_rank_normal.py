#!/usr/bin/env python
# note need to run viewer with python2!!!

from cmath import e
import os
import sys
import csv
import numpy as np
from IPython import embed
from PIL import Image
GRID_RESOLUTION = 0.1
CLEARANCE_THRESH = 0.5/GRID_RESOLUTION
GRID_SIZE_IN_M = 6
# myargv = rospy.myargv(argv=sys.argv)
# scene = myargv[1]
# driving = myargv[2]
# if driving=="true" or driving =="True":
#     OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_feb_6/driving/"
# else:
#     OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/irl_feb_6/rl/"
OUT_DIR = "/home/catkin_ws/src/habitat_ros_interface/data/dataset_full_ep/irl_jul_19_5/"
METRIC_CSV = OUT_DIR + "metrics_data.csv"
FIXED_LEN = 10

def auto_pad_future(traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = FIXED_LEN
        if traj.shape[0] >= fixed_len:
            traj = traj[:fixed_len, :]
            return traj
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = fixed_len - traj.shape[0]
        
        pad_list = []
        for i in range(int(np.ceil(pad_len))):
            if (i < pad_len):
                # pad_list.append([(traj[-1,0]-1), traj[-1,1]])
                pad_list.append([traj[-1,0], traj[-1,1]])
            else:
                pad_list.append([np.NaN, np.NaN])
        # print(pad_list)
        pad_array = np.array(pad_list[:pad_len])
        if pad_len>0:
            output = np.vstack((traj, pad_array))
        else:
            output = traj
        return output

def transpose_traj(traj):
    for i in range(traj.shape[0]):
        temp = traj[i,0] 
        traj[i,0]= traj[i,1]
        traj[i,1] = temp 
    return traj

def check_neighbor(first, second):
    actions = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1]])
    possible_neighbors = list(first) + actions
    for neighbor in possible_neighbors:
        if (second==neighbor).all():
            return True
    return False
    
def is_valid_traj(traj):
    i = 0
    while i < traj.shape[0]-1:
        if not check_neighbor(traj[i], traj[i+1]):
            return False
        i = i+1
    return True
def get_traj_length_unique(traj):
    lengths = []
    traj_list = []
    for j in range(traj.shape[0]):
        # if list(traj[j]) not in traj_list:
        if True:
            traj_list.append([traj[j][0], traj[j][1]])
   
    return traj.shape[0], np.array(traj_list)

def traj_interp(c):
    d = c.astype(int)
    iter = d.shape[0] - 1
    added = 0
    i = 0
    while i < iter:
        while np.sqrt((d[i+added,0]-d[i+1+added,0])**2 + (d[i+added,1]-d[i+1+added,1])**2) > np.sqrt(1):
            d = np.insert(d, i+added+1, [0, 0], axis=0)
            if d[i+added+2, 0] - d[i+added, 0] > 0:
                d[i+added+1, 0] = d[i+added, 0] + 1
                d[i+added+1, 1] = d[i+added, 1]
            elif d[i+added+2, 0] - d[i+added, 0] < 0:
                d[i+added+1, 0] = d[i+added, 0] - 1
                d[i+added+1, 1] = d[i+added, 1]
            else:
                d[i+added+1, 0] = d[i+added, 0]
                if d[i+added+2, 1] - d[i+added, 1] > 0:
                    d[i+added+1, 1] = d[i+added, 1] + 1
                elif d[i+added+2, 1] - d[i+added, 1] < 0:
                    d[i+added+1, 1] = d[i+added, 1] - 1
                else:
                    d[i+added+1, 1] = d[i+added, 1]
            added += 1
        i += 1
    # connected_map = np.zeros((32, 32))
    # for i in range(len(d)):
    #     connected_map[int(d[i,1])+1, int(d[i,0])+1] = 1
    if not is_valid_traj(d):
        print(d)
    # print(c)
    # print(d)
    # print("Valid traj? ", is_valid_traj(d), d.shape, c.shape)
    return d

def check_same_episode(path1, path2):
    img1 = np.array(Image.open(path1+"/0/grid_map.png"))[:,:,0:3].T
    img2 = np.array(Image.open(path2+"/0/grid_map.png"))[:,:,0:3].T
    with open(path1+"/0/goal.npy", 'rb') as f:
        goal1 = np.load(f)
    with open(path2+"/0/goal.npy", 'rb') as f:    
        goal2 = np.load(f)
    if (img1 == img2).all() and (goal1 == goal2).all():
        return True
    return False

class Rank():
    def __init__(self):
        self.metrics = []
        rows = []
        self.current_metric = None
        metric = {}
        with open(METRIC_CSV, mode='r', newline='') as file:
            # Create a DictReader object
            csv_reader = csv.DictReader(file)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Print each row as a dictionary
                print(row)
                self.metrics.append(row)
        # keys = rows[0].keys()
        # embed()
        # for row in rows[1:]:
        #     i =0
        #     for key in keys:
        #         metric[key] = row[i]
        #         i+=1
        #     self.metrics.append(metric)
        #     metric = {}
        # print("Self metrics is ", self.metrics)

        self.row_counter = 0

    def get_rank(self, row_counter):
        metric = self.metrics[row_counter]
        
        return metric
    

if __name__ == "__main__":
    rank_class = Rank()
    max_num = 0
    for foldername in os.listdir(OUT_DIR):
        if foldername[:5] != "demo_":
            continue
        number_str = "_"
        valid = False
        m = foldername[5:]
        max_num = max(max_num,int(m))
    row_num = 0
    ep_list = []
    dist_list = []
    for demo_num in range(max_num):
        foldername = "demo_" + str(demo_num)
        if not os.path.isdir(OUT_DIR + foldername):
            continue
        print("Processing folder ", foldername)
        full_path_now = OUT_DIR + foldername
        with open(full_path_now + "/ep_count.txt", 'r') as f:
            ep_num = f.read()
            f.close()
        ep_num = int(ep_num)
        metrics = rank_class.get_rank(row_counter=row_num)
        print("Metrics are ", metrics)  
        humanposes = []
        robotposes = []
        distances = []    

        folders = os.listdir(full_path_now)

        for folder_path in folders:
            full_path = OUT_DIR + foldername + "/" + folder_path
            # print("Full path is ", full_path, len(str(folder_path)))
            if len(folder_path) > 4:
                continue
            
            #### Find out the min dist between human and robot poses, go over all folders indide this folder
            with open(full_path+"/human_past_traj.npy", 'rb') as f:
                full_traj = np.load(f)
            # if len(full_traj) == 0:
            #     with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
            #         full_traj = np.load(f)
            full_traj = traj_interp(full_traj)
            length, human_past_traj = get_traj_length_unique(full_traj)
            

            with open(full_path+"/robot_past_traj.npy", 'rb') as f:
                full_traj = np.load(f)
            # if len(full_traj) == 0:
            #     with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
            #         full_traj = np.load(f)
            full_traj = traj_interp(full_traj)
            length, robot_past_traj = get_traj_length_unique(full_traj)
            

            # with open(full_path+"/traj.npy", 'rb') as f:
            #     full_traj = np.load(f)
            # # full_traj = np.array(traj_interp(full_traj), np.int)
            # # print("Valid full traj? ", is_valid_traj(full_traj))
            # len, robot_traj = get_traj_length_unique(full_traj)
            
            # # past_ind= np.where(robot_traj == robot_past_traj[-1])
            # past_ind = 0
            # for i in range(len[0]):
            #     if (robot_traj[i] == robot_past_traj[-1]).all():
            #         past_ind = i
            # temp = robot_traj
            # robot_traj = robot_traj[past_ind:]
            
            # len = robot_traj.shape[0]
            # if len == 1:
            #     robot_traj = np.vstack((robot_traj, np.array([robot_traj[0,0]-1, robot_traj[0,1]])))
            # # print("Fial traj ", is_valid_traj(robot_traj))
            # if (not is_valid_traj(robot_traj)):
            #     print(past_ind)
            #     print(robot_past_traj[-1], robot_traj, len)
            #     print(temp)
            # robot_past_traj = transpose_traj(robot_past_traj)
            robot_pos = np.array([robot_past_traj[-1,0], robot_past_traj[-1,1]])
            human_pos = np.array([human_past_traj[-1,0], human_past_traj[-1,1]])
            robotposes.append(robot_pos)
            humanposes.append(human_pos)
            try:
                distances.append(np.linalg.norm(robot_pos - human_pos)*GRID_RESOLUTION)
            except:
                print("Robot pos is ", robot_pos)
                print("Human pos is ", human_pos)
        # print("Distances are ", distances)
        if distances == []:
            print("No distances found for ", foldername)
            continue
        rank = min(distances)
        # if ep_num == 1:
        row_num += 1
        # else:
            # print("New Episode found ", foldername)
        humanposes = []
        robotposes = []
        distances = []
        success = metrics["social_nav_to_pos_success"] == "True" or metrics["social_nav_to_pos_success"] == "TRUE"
        if len(ep_list) == 0:
            ep_list.append([{'fold_name': foldername, 'dist': rank, 'num_steps': int(metrics['num_steps']), 'succ': success}])
        else:
            already_in = False
            for bin in ep_list:
                if check_same_episode(full_path_now, OUT_DIR + bin[0]["fold_name"]):
                    bin.append({'fold_name': foldername, 'dist': rank, 'num_steps': int(metrics['num_steps']), 'succ': success})
                    already_in = True
                    break
            if not already_in:
                ep_list.append([{'fold_name': foldername, 'dist': rank, 'num_steps': int(metrics['num_steps']), 'succ': success}])          
        # with open(full_path_now + "/rank.txt", 'w') as f:
        #     f.write(str(rank))
        #     f.close()
        # with open(full_path_now+"/metrics.csv", "w", newline="") as fp:
        # # Create a writer object
        #     writer = csv.DictWriter(fp, fieldnames=metrics.keys())
        #     writer.writeheader()
        #     writer.writerow(metrics)
        # robot_traj = auto_pad_future(robot_traj[:, :2])
        print("Ep list is ", ep_list)

    embed()
    for eps in ep_list:
        min_steps = 1000
        for ep in eps:
            if ep['num_steps'] < min_steps:
                min_steps = ep['num_steps']
        for ep in eps:
            new_rank = ep['succ']*min_steps/ep['num_steps']
            print("New rank is ", new_rank)
            with open(OUT_DIR + ep['fold_name'] + "/new_rank.txt", 'w') as f:
                f.write(str(new_rank))
                f.close()

        
        # print("Rank is ", rank.get_rank(foldername))