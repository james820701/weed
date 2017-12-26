import keras
from keras.models import Model
from keras.layers import Dense
import cv2
from keras.utils import to_categorical
import sys, os
import numpy as np
import glob
import ipdb

def build_model(num_classes=12):
	#model = keras.applications.vgg16.VGG16(weights=None)
	model = keras.applications.inception_v3.InceptionV3()
	input = model.layers[0].output	
	x = model.layers[-2].output
	x = Dense(num_classes, activation='softmax', name='seed_prediction')(x)
	model = Model(inputs=input, outputs=x)
	model.summary()

	return model


def load_image_and_classes(image_dir, total_files, num_classes):
	i = 0
	images = np.zeros((total_files, 299, 299, 3))
	labels = np.zeros((total_files, 1))
	j = 0
	for classes in os.listdir(image_dir):
		image_path = os.path.join(image_dir, classes)
		files = glob.glob(image_path + "/*.png")
		for name in files:
			image = cv2.imread(name)
			image = cv2.resize(image, (299,299))
			# print(name)
			# print(image.shape)
			images[j,:,:,:] = image
			labels[j, 0] = i
			j += 1

		i += 1
	labels = to_categorical(labels, num_classes)
	return images, labels

model = build_model(num_classes=12)
total_train_files = 0
total_valid_files = 0
train_image_dir = sys.argv[1]
valid_image_dir = sys.argv[2]
weights = sys.argv[3]

num_classes = 0
for classes in os.listdir(train_image_dir):
	train_image_path = os.path.join(train_image_dir, classes)
	total_train_files += len(os.listdir(train_image_path))
	num_classes += 1

for classes in os.listdir(valid_image_dir):
	valid_image_path = os.path.join(valid_image_dir, classes)
	total_valid_files += len(os.listdir(valid_image_path))


train_images, train_labels = load_image_and_classes(train_image_dir, total_train_files, num_classes)
valid_images, valid_labels = load_image_and_classes(valid_image_dir, total_valid_files, num_classes)
#ipdb.set_trace()
loss=keras.losses.categorical_crossentropy
sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=False)
if weights != "None":
	model.load_weights(weights, by_name=True)
model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
model.fit(train_images, train_labels, shuffle=True, batch_size=32, epochs=6, validation_data=(valid_images, valid_labels))
model.save("seed_prediction.h5")