import os, sys


directory = sys.argv[1]

for classes in os.listdir(directory):
	if " " in classes:
		path = os.path.join(directory, classes)
		ori_path = path.replace(" ", "\\ ")
		new_path = path.replace(" ", "-")

		os.system("mv " + ori_path + " " + new_path)