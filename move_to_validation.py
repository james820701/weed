import sys, os
import numpy as np
import glob

train_dir = sys.argv[1]
valid_dir = sys.argv[2]


for classes in os.listdir(train_dir):
	image_dir = os.path.join(train_dir, classes)
	files = glob.glob(image_dir + "/*.png")
	np.random.seed(10)
	sample = np.random.choice(len(files), int(0.05*len(files)))
	savedir = os.path.join(valid_dir, classes)
	for i in sample:
		file = files[i]
		os.system("mv " + file + " " + savedir)
	
