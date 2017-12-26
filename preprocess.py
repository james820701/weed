import cv2
from matplotlib import pyplot as plt
import sys, os
import numpy as np




train_dir = sys.argv[1]
save_dir = sys.argv[2]

def preprocess(image_dir, save_dir):
	for file in os.listdir(image_dir):
		filename = os.path.join(image_dir, file)
		image = cv2.imread(filename)
		b = image[:,:,0] < image[:,:,1]
		b2 = image[:,:,2] < image[:,:,1]
		c = np.zeros(image.shape)
		c[:,:,0] = np.multiply(image[:,:,0], b)
		c[:,:,1] = np.multiply(image[:,:,1], b)
		c[:,:,2] = np.multiply(image[:,:,2], b)
		c[:,:,0] = np.multiply(c[:,:,0], b2)
		c[:,:,1] = np.multiply(c[:,:,1], b2)
		c[:,:,2] = np.multiply(c[:,:,2], b2)
		
		savedname = os.path.join(save_dir, file)
		cv2.imwrite(savedname, c)

for classes in os.listdir(train_dir):
	path = os.path.join(train_dir, classes)
	save_path = os.path.join(save_dir, classes)
	preprocess(path, save_path)
		