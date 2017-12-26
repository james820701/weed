import os, sys


directory = sys.argv[1]
dir2 = sys.argv[2]

for classes in os.listdir(directory):
	path = os.path.join(dir2, classes)
	os.system("mkdir " + path)