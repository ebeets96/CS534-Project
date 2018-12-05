import numpy as np
import os
import sys,getopt
import random
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave

epochs = input("Number of epochs: ")
batchSize = input("Batch size: ")
steps_per_epoch = input("Steps per epoch: ")
trainFile = 'flickr_mac_cropped'
testFile = 'test_cropped_mac'
resultName = "result_e{0}_b{1}_s{2}_tushar".format(epochs, batchSize, steps_per_epoch)
resultFile = "{0}".format(resultName)

# Get images
print("Load training images");
trainImages = []
i = 0
for filename in os.listdir(trainFile+"/directory"):
	try:
		trainImages.append(img_to_array(load_img(trainFile+"/directory/"+filename)))
		i = i + 1
		if(i == 1000):
			break
	except:
		print("skipping " + trainFile+"/directory/"+filename)

trainImages = np.array(trainImages, dtype=float)
# Set up training and test data
number_of_images = len([name for name in os.listdir(trainFile) if os.path.isfile(os.path.join(trainFile, name))])
split_size = int(0.05*number_of_images)
epoch_steps = (number_of_images - split_size)
# split_set = getSplit(split_size)
trainSet = trainImages
trainSet = trainSet*1.0/255

# Generate training data
def imageGenerator(batchSize):
	for batch in datagen.flow(trainSet, batch_size=batchSize):
	# for batch in datagen.flow_from_directory(trainFile, batch_size=batchSize, class_mode=None):
		lab_batch = rgb2lab(batch)
		train_batch = lab_batch[:,:,:,0]
		test_batch = lab_batch[:,:,:,1:]
		test_batch = test_batch / 128
		yield (train_batch.reshape(train_batch.shape+(1,)), test_batch)


print("\n Create Data Generator")
# Image transformer
datagen = ImageDataGenerator(
		validation_split=0.05,
        shear_range=0.1,
        zoom_range=0.02,
        rotation_range=10,
        horizontal_flip=False)

print("Create Network Model")

#Neural Net model
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.compile(optimizer='rmsprop', loss='mse')
# Train model
# Don't change steps_per_epoch. Vary the epochs only
model.fit_generator(imageGenerator(batchSize), steps_per_epoch=steps_per_epoch, epochs=epochs)

# Save model
model_json = model.to_json()
with open(resultName+".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(resultName+".h5")

# model.load_weights("colorModelWeights.h5")

# Test images
# Xtest = rgb2lab(1.0/255*split_set)[:,:,:,0]
# Xtest = Xtest.reshape(Xtest.shape+(1,))
# Ytest = rgb2lab(1.0/255*split_set)[:,:,:,1:]
# Ytest = Ytest / 128
# print(model.evaluate(Xtest, Ytest, batch_size=batchSize))

# Load black and white images from the test/ folder
colorImages = []
print("Load test images")
for filename in os.listdir(testFile):
	try:
		file = testFile+"/"+filename
		print(file)
		colorImages.append(img_to_array(load_img(file)))
	except:
		print("Skipping {0}".format(filename))

colorImages = np.array(colorImages, dtype=float)
colorImages = rgb2lab(1.0/255*colorImages)[:,:,:,0]
colorImages = colorImages.reshape(colorImages.shape+(1,))

#create result folder
if not os.path.exists(resultFile):
    os.makedirs(resultFile)

# Test model
print("Test model: " + str(len(colorImages)))
output = model.predict(colorImages)
output = output * 128
# Output colorizations
for i in range(len(output)):
	print(resultFile+"/img_"+str(i)+".png")
	cur = np.zeros((256, 256, 3)) #change to 256,256
	cur[:,:,0] = colorImages[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave(resultFile+"/img_"+str(i)+".png", lab2rgb(cur))
