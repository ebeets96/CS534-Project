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


trainFile ='train_cropped/'
testFile = 'test_cropped/'
resultFile = 'result/'
epochs = 1000 #1000


# Get images
print("Load training images");
trainImages = []
for filename in os.listdir(trainFile):
    trainImages.append(img_to_array(load_img(trainFile+filename)))
trainImages = np.array(trainImages, dtype=float)
# Set up training and test data
split = int(0.95*len(trainImages))
trainSet = trainImages[:split]
trainSet = trainSet*1.0/255

print("\n Create Data Generator")
# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.02,
        rotation_range=10,
        horizontal_flip=False)


# Generate training data
batchSize = 10
def imageGenerator(batchSize):
    for batch in datagen.flow(trainSet, batch_size=batchSize):
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
model.fit_generator(imageGenerator(batchSize), steps_per_epoch=3, epochs=epochs)

# Save model
model_json = model.to_json()
with open("colorModel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("colorModelWeights.h5")

# model.load_weights("colorModelWeights.h5")

# Test images
Xtest = rgb2lab(1.0/255*trainImages[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*trainImages[split:])[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batchSize))

# Load black and white images from the test/ folder
colorImages = []
a = 0;
for filename in os.listdir(testFile):
	colorImages.append(img_to_array(load_img(testFile+filename)))
	a = a + 1
	if(a>100):
		break

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
