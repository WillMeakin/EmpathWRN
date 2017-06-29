import csv
import numpy as np
from keras.utils import to_categorical
from keras.backend import image_data_format
import sys

#Assumed csv format: first row has column keys
#Column: 0=int label, 1=int data list (<space> delimiter), 2=use (e.g. Training/Validation/Test)
def readCSV(fileName, rowDim, channels, useIDs=('Training', 'Validation', 'Test')):
	print('Parsing: ', fileName, '...')
	trainLabels = []
	trainData = []
	validationLabels = []
	validationData = []
	testLabels = []
	testData = []

	if image_data_format() == 'channels_last':
		shapeTuple = (rowDim, rowDim, channels)
	elif image_data_format() == 'channels_first':
		shapeTuple = (channels, rowDim, rowDim)
	else:
		print("channel not first or last, aborting")
		sys.exit()

	with open(fileName) as dataFile:
		reader = csv.reader(dataFile)
		next(reader) #skip first row with column keys
		for row in reader:
			if reader.line_num % 1000 == 0:
				print('row: ', reader.line_num)

			#img data
			data = np.fromstring(row[1], dtype=int, sep=' ')
			data = np.reshape(data, shapeTuple)

			if row[2] == useIDs[0]: #training
				trainLabels.append(np.asarray([int(row[0])]))
				trainData.append(data)
			elif row[2] == useIDs[1]: #validation
				validationLabels.append(np.asarray([int(row[0])]))
				validationData.append(data)
			elif row[2] == useIDs[2]: #testing
				testLabels.append(np.asarray([int(row[0])]))
				testData.append(data)
			else:
				print("Err: don't know what this row is for; check useIDs arg", row)

	#TODO: should valid/test labels use train labels' unique length? In case they don't have all types in them.
	#reformat for keras (and type->np.ndarray)
	trainLabels = to_categorical(trainLabels, num_classes=len(np.unique(trainLabels)))
	validationLabels = to_categorical(validationLabels, num_classes=len(np.unique(validationLabels)))
	testLabels = to_categorical(testLabels, num_classes=len(np.unique(testLabels)))

	trainData = np.asarray(trainData)
	validationData = np.asarray(validationData)
	testData = np.asarray(testData)

	print('channelMode: ', image_data_format(), ' trainData shape:', trainData.shape)
	#print(trainData)

	#Normalise #TODO: meanstd?
	trainData = trainData.astype('float32')
	validationData = validationData.astype('float32')
	testData = testData.astype('float32')
	trainData /= 255 #TODO: why not .0?
	validationData /= 255
	testData /= 255

	print("Parsing finished.")

	trainDSize = sys.getsizeof(trainData)
	validDSize = sys.getsizeof(validationData)
	testDSizee = sys.getsizeof(testData)
	trainLSize = sys.getsizeof(trainLabels)
	validLSize = sys.getsizeof(validationLabels)
	testLSizee = sys.getsizeof(testLabels)
	print("traindatsize: ", trainDSize/1000000)
	print("validatisize: ", validDSize/1000000)
	print("testdatasize: ", testDSizee/1000000)
	print("trainLabsize: ", trainLSize/1000000)
	print("valiLabisize: ", validLSize/1000000)
	print("testLabasize: ", testLSizee/1000000)
	print("total: MB", (trainDSize + validDSize + testDSizee + trainLSize + validLSize + testLSizee)/1000000)

	return trainLabels, trainData, \
		   validationLabels, validationData, \
		   testLabels, testData