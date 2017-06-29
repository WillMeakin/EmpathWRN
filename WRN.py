from keras.models import Model
from keras.layers import Conv2D, Dropout, Input, Activation, BatchNormalization, add, AveragePooling2D, Flatten, Dense
from keras.backend import image_data_format, shape
from keras import backend as K

def resUnit(inTensor, nFilters, stride, axis, identityConv=False):

	x = inTensor

	if identityConv: #reshape identity to allow merging. Downsamples on 2nd and 3rd group
		inTensor = Conv2D(nFilters,
						  (1, 1),#Kernel dim
						  strides=(stride, stride),
						  padding='same')(inTensor)

	x = BatchNormalization(axis=axis)(x)
	x = Activation('relu')(x)
	x = Conv2D(nFilters,
			   (3, 3), #Kernel dim
			   strides=(stride, stride),
			   padding='same')(x)

	x = BatchNormalization(axis=axis)(x)
	x = Activation('relu')(x)
	x = Dropout(0.3)(x)
	x = Conv2D(nFilters,
			   (3, 3), #Kernel dim #stride default 1, don't downsample second
			   padding='same')(x)

	x = add([x, inTensor]) #Merge identiy tensor
	return x

def makeModel(inShape, nClasses, n=4, k=4): #k is width multiplier, n is resnet blocks per group n={4=WRN-28-k, 12=WRN-40-k}

	axis = -1 #Used for BatchNorm, detecting output shape change
	if image_data_format() == 'channels_first':
		axis = 1
	elif image_data_format() == 'channels_last':
		axis = 3

	inputs = Input(shape=inShape)
	print('inShape:', inShape)

	#Init convolution
	x = Conv2D(	16, #Number of filters
				(3, 3))(inputs) #Kernel dim (stride default 1)

	for groupi in range(3): #groups 2, 3, 4 in iters 0, 1, 2
		for resUniti in range(n):
			print('width:', 16 * k * 2 ** groupi)
			if groupi == 0 and resUniti == 0: #init 16 -> 16k filters
				#increases output dim from first conv2d to merge with 16k, but no downsample
				x = resUnit(x, 16*k*2**groupi, 1, axis, identityConv=True)
			elif resUniti == 0: #16k->32k and 32k->64k nFilters. 32->16 and 16->8 imgdim. (Cifar)
				#increases output dim and downsamples at start of group
				x = resUnit(x, 16*k*2**groupi, 2, axis, identityConv=True)
			else: #normal resUnit
				x = resUnit(x, 16*k*2**groupi, 1, axis)

	x = BatchNormalization(axis=axis)(x)
	x = Activation('relu')(x)
	x = AveragePooling2D(8, 1)(x)

	x = Flatten()(x)
	predictions = Dense(nClasses, activation='softmax')(x)

	return Model(inputs=inputs, outputs=predictions)