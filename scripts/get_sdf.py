import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from IPython import embed
import argparse
import sdf_grid.srv from srv
PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-s', '--scene', default="17DRP5sb8fy", type=str, help='scene')
ARGS = PARSER.parse_args()
scene = ARGS.scene


map_file = "./maps/resolution_"+scene+"_0.025.pgm"
dist_map_file = "./maps/sdf_resolution_"+scene+"_0.025.pgm"
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
