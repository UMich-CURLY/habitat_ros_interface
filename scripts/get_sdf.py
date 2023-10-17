import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from IPython import embed
import argparse
import yaml
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
ARGS = PARSER.parse_args()
scene = ARGS.scene


map_file = "./maps/resolution_"+scene+"_0.025.pgm"
map_yaml_file = map_file[:-3]+"yaml"
with open(map_yaml_file,'r') as file:
    old_config = yaml.load(file, Loader=yaml.FullLoader)
dist_map_file = "./maps/sdf_resolution_"+scene+"_0.025.pgm"
old_config["image"] = "sdf_"+old_config["image"]
new_yaml_file = dist_map_file[:-3]+"yaml"
with open(new_yaml_file,'w+') as file: 
    yaml.dump(old_config, file)
img = cv.imread(map_file)
assert img is not None, "file could not be read, check with os.path.exists()"
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imagem = cv.bitwise_not(gray)
ret, thresh = cv.threshold(imagem,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
cv.imwrite(dist_map_file,dist_transform)


a = np.where(dist_transform <18, dist_transform, 0)
b = np.where(a >5, a, 0)
d = np.where(b!=0)
candidate_indices = []
x_window = 20
y_window = 20
for i in range(d[0].shape[0]):
    x_index = d[0][i]
    y_index = d[1][i]
    right_max = np.max(dist_transform[x_index:x_index+x_window, y_index:y_index+y_window])
    left_max = np.max(dist_transform[x_index-x_window:x_index, y_index-y_window:y_index])
    if (right_max > 20 and right_max >20):
        candidate_indices.append([d[0][i], d[1][i]])
        candidate_indices.append([d[0][i]+x_window, d[1][i]+y_window])
candidate_indices = np.array(candidate_indices)
x = np.array(candidate_indices[:,0])
y = np.array(candidate_indices[:,1])
candidate_indices = (x,y)
fig = np.zeros(dist_transform.shape)
gray[candidate_indices] = 0
cv.imwrite("trying.pgm", gray)
