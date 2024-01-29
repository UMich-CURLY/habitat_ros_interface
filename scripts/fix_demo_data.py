import os 
import argparse
import numpy as np
from IPython import embed
import cv2
PARSER = argparse.ArgumentParser(description=None)

PARSER.add_argument('-d', '--dataset', default="mp3d", type=str, help='dataset')
ARGS = PARSER.parse_args()
dataset = ARGS.dataset

DATA_PATH = "./data/datasets/irl_jan_12/driving"
content_path = "/home/catkin_ws/src/habitat_ros_interface/data/datasets/pointnav/mp3d/v1/test/images"
invalid_scenes = []

def check_neighbor(first, second):
    actions = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1]])
    possible_neighbors = list(first) + actions
    for neighbor in possible_neighbors:
        if (second==neighbor).all():
            return True
    return False
    
def is_valid_traj(traj):
    i = 0
    while i < len(traj)-1:
        if not check_neighbor(traj[i], traj[i+1]):
            return False
        i = i+1
    return True

def traj_interp(c):
    d = c.astype(int)
    iter = len(d) - 1
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
    connected_map = np.zeros((32, 32))
    for i in range(len(d)):
        connected_map[int(d[i,1])+1, int(d[i,0])+1] = 1
    return d

for foldername in os.listdir(DATA_PATH):
    folder_path = DATA_PATH+"/"+foldername
    files = os.listdir(folder_path)
 
    scene = None
    for j in files:
        if "json.gz" in j:
            scene = j[:-9]
            break
    ### Copy from the created test folder match scene names
    if scene is None:
        continue
    for content in os.listdir(content_path):
        if scene == content:
            __ = os.system('cp '+ content_path+"/"+scene+"/semantic_img.png " + folder_path)
    try:
        traj_data = np.load(folder_path+"/trajectory.npy")
    except:
        list_traj_data = np.load(folder_path+"/trajectory.npy", allow_pickle= True)
        list_traj_data = list_traj_data[1:]
        traj_data = np.zeros([list_traj_data.shape[0],2,2])
        for i in range(list_traj_data.shape[0]):
            traj_data[i,0,:] = list_traj_data[i][0]
            traj_data[i,1,:] = list_traj_data[i][1]
    robot_data = np.zeros([traj_data.shape[0], 2])
    for i in range(traj_data.shape[0]):
        robot_data[i] = np.array(traj_data[i,0])
    new_robot_data = traj_interp(robot_data)
    traj_data = traj_data.astype(int)
    print("Initial ", traj_data, is_valid_traj(traj_data[:,0,:]))
    semantic_img = cv2.imread(content_path+"/"+scene+"/semantic_img.png") 
    for i in range(traj_data.shape[0]):
        semantic_img[traj_data[i,0,1], traj_data[i,0,0]] = [0,0,0]
        semantic_img[traj_data[i,1,1], traj_data[i,1,0]] = [255,0,0]

    cv2.imwrite(folder_path+ "/traj_not_fixed_feat.png", semantic_img)
    for i in range(len(new_robot_data)):
        if (traj_data[i][:][0] == new_robot_data[i]).all():
            continue
        else:
            traj_data = np.concatenate((traj_data[:i,:,:], np.reshape([new_robot_data[i,:], traj_data[i,1,:]], (1,2,2)), traj_data[i:,:,:]), axis=0)
    print("After robot ", is_valid_traj(traj_data[:,0,:]))
    human_data = np.zeros([traj_data.shape[0], 2])
    for i in range(traj_data.shape[0]):
        human_data[i] = np.array(traj_data[i,1])
    new_human_data = traj_interp(human_data)
    for i in range(len(new_human_data)):
        if (traj_data[i][:][1] == new_human_data[i]).all():
            continue
        else:
            traj_data = np.concatenate((traj_data[:i,:,:], np.reshape([traj_data[i,0,:], new_human_data[i,:]], (1,2,2)), traj_data[i:,:,:]), axis=0)
    print("Final ", is_valid_traj(traj_data[:,0,:]))
    if not (is_valid_traj(traj_data[:,0,:]) or not is_valid_traj(traj_data[:,1,:])):
        embed()
    with open(folder_path+"/traj_fixed.npy", 'wb') as f:
        np.save(f, np.array(traj_data))
    traj_data = traj_data.astype(int)
    semantic_img = cv2.imread(content_path+"/"+scene+"/semantic_img.png") 
    for i in range(traj_data.shape[0]):
        semantic_img[traj_data[i,0,0], traj_data[i,0,1]] = [0,0,0]
        semantic_img[traj_data[i,1,0], traj_data[i,1,1]] = [255,0,0]

    cv2.imwrite(folder_path+ "/traj_fixed_feat.png", semantic_img)
    robot_traj_data = np.load(folder_path+"/robot_traj.npy")
    new_robot_data = traj_interp(robot_traj_data)
    with open(folder_path+"/robot_traj_fixed.npy", 'wb') as f:
        np.save(f, np.array(new_robot_data))
    human_traj_data = np.load(folder_path+"/human_traj.npy")
    new_human_data = traj_interp(human_traj_data)
    with open(folder_path+"/human_traj_fixed.npy", 'wb') as f:
        np.save(f, np.array(new_human_data))
    human_past_traj_data = np.load(folder_path+"/human_past_traj.npy")
    past_traj_data = traj_interp(human_past_traj_data)
    with open(folder_path+"/human_past_traj_fixed.npy", 'wb') as f:
        np.save(f, np.array(past_traj_data))
    if len(past_traj_data) == 0:
        past_traj_data = np.array([new_human_data[0]])
    semantic_img = cv2.imread(content_path+"/"+scene+"/semantic_img.png") 
    semantic_img[new_robot_data[:,0], new_robot_data[:,1]] = [0,0,0]
    semantic_img[past_traj_data[:,0], past_traj_data[:,1]] = [255,0,0]
    semantic_img[new_human_data[:,0], new_human_data[:,1]] = [0,255,0]
    cv2.imwrite(folder_path+ "/traj_fixed_trajs.png", semantic_img)
    if foldername == "demo_36":
        embed()
#     if (x == 0):
#         __ = os.system('python ./scripts/get_topdown_map_rl.py --scene '+ scene )
#         __ = os.system('python ./maps/get_outline.py --scene '+ scene)
#         __ = os.system('python ./scripts/get_sdf.py --scene '+ scene)
#     else:
#         invalid_scenes.append(foldername)
# print("These scenes did not have a solution ", invalid_scenes)