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



#Cmdline arguments
# def main(argv):
# 	try:
# 		opts,args =  getopt.getopt(argv,"e:")#in:out:test:")
# 		print 'try block ran'
# 	except getopt.GetoptError:
# 		print 'colorizer.py -epoch <integer> -in <trainset> -test <testset> -out<result>'
# 		sys.exit(2)

# 	for opt, arg in opts:
# 		if opt == '-e':
# 			epochNum = arg
# 			print 'epochNum is set to '+arg

# if __name__ == "__main__":
# 	epochNum = 0
# 	main(sys.argv[1:])
# 	print 'main is being run'

epochs = 40
batchSize = 20
steps_per_epoch = 4000
trainFile ='flickr3_cropped/'
testFile = 'test_cropped/'
resultFile = 'result_e{0}_b{1}_s{2}/'.format(epochs, batchSize, steps_per_epoch)

if not os.path.exists(resultFile):
    os.makedirs(resultFile)

# def create_mini_training_set (number_of_images, split_size):
# 	trainImages = []
# 	skipped = 0
# 	for filename in os.listdir(trainFile):
# 		#skip split files
# 		if(skipped < split_size):
# 			skipped = skipped + 1
# 			continue
#
# 		trainImages.append(img_to_array(load_img(trainFile+filename)))
# 		if(len(trainImages) == number_of_images):
# 			trainImages = np.array(trainImages, dtype=float)
# 			yield trainImages
# 			trainImages = []
#
# 	if (len(trainImages) > 0):
# 		trainImages = np.array(trainImages, dtype=float)
# 		yield trainImages
#
# def getSplit (split_size):
# 	splitImages = []
# 	for filename in os.listdir(trainFile):
# 		splitImages.append(img_to_array(load_img(trainFile+filename)))
# 		if(len(splitImages) == split_size):
# 			splitImages = np.array(splitImages, dtype=float)
# 			return splitImages

# Get images
# print("Load training images");
# trainImages = []
# i = 0
# for filename in os.listdir(trainFile):
# 	trainImages.append(img_to_array(load_img(trainFile+filename)))
# 	i = i + 1
# 	if(i == number_of_images):
# 		break

# trainImages = np.array(trainImages, dtype=float)
# Set up training and test data
# number_of_images = len([name for name in os.listdir(trainFile) if os.path.isfile(os.path.join(trainFile, name))])
# split_size = int(0.05*number_of_images)
#epoch_steps = (number_of_images - split_size)
# split_set = getSplit(split_size)
# trainSet = trainImages[:split]
# trainSet = trainSet*1.0/255

print("\n Create Data Generator")
# Image transformer
datagen = ImageDataGenerator(
		validation_split=0.05,
        shear_range=0.1,
        zoom_range=0.02,
        rotation_range=10,
        horizontal_flip=False)

# Generate training data
def imageGenerator(batchSize):
	# for trainImages in create_mini_training_set(500, split_size):
	# 	print("next mini")
	# 	trainSet = trainImages*1.0/255
	#for batch in datagen.flow(trainSet, batch_size=batchSize):
	for batch in datagen.flow_from_directory(trainFile, batch_size=batchSize, class_mode=None):
		lab_batch = rgb2lab(batch)
		train_batch = lab_batch[:,:,:,0]
		test_batch = lab_batch[:,:,:,1:]
		test_batch = test_batch / 128
		yield (train_batch.reshape(train_batch.shape+(1,)), test_batch)


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
with open("colorModel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("colorModelWeights.h5")

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
	colorImages.append(img_to_array(load_img(testFile+filename)))

colorImages = np.array(colorImages, dtype=float)
colorImages = rgb2lab(1.0/255*colorImages)[:,:,:,0]
colorImages = colorImages.reshape(colorImages.shape+(1,))

# Test model
print("Test model")
output = model.predict(colorImages)
output = output * 128
# Output colorizations
for i in range(len(output)):
	cur = np.zeros((256, 256, 3)) #change to 256,256
	cur[:,:,0] = colorImages[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave( resultFile+"img_"+str(i)+".png", lab2rgb(cur))
