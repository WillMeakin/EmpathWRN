import csv, sys, os, cv2
import numpy as np
from keras.utils import to_categorical
from keras.backend import image_data_format

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

	shapeTuple = getShapeTuple(rowDim, channels)

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
	# print(np.shape(trainLabels))
	# print(trainLabels)

	#reformat for keras (and type->np.ndarray)
	trainLabels = to_categorical(trainLabels, num_classes=len(np.unique(trainLabels)))
	validationLabels = to_categorical(validationLabels, num_classes=len(np.unique(validationLabels)))
	testLabels = to_categorical(testLabels, num_classes=len(np.unique(testLabels)))

	trainData = np.asarray(trainData)
	validationData = np.asarray(validationData)
	testData = np.asarray(testData)

	print('channelMode: ', image_data_format(), ' trainData shape:', trainData.shape)

	trainData = trainData.astype('float32')
	validationData = validationData.astype('float32')
	testData = testData.astype('float32')
	trainData /= 255 #TODO: why not .0?
	validationData /= 255
	testData /= 255

	ferMean = np.mean(np.concatenate((trainData, testData, validationData)))
	ferStd = np.std(np.concatenate((trainData, testData, validationData)))
	print('ferMean:', ferMean)
	print('ferStd:', ferStd)

	#Normalise
	for imgi in range(len(trainData)):
		trainData[imgi] = (trainData[imgi]-ferMean)**2/ferStd
	for imgi in range(len(testData)):
		testData[imgi] = (testData[imgi] - ferMean)**2 / ferStd
	for imgi in range(len(validationData)):
		validationData[imgi] = (validationData[imgi] - ferMean)**2 /ferStd

	print("Parsing finished.")

	# trainDSize = sys.getsizeof(trainData)
	# validDSize = sys.getsizeof(validationData)
	# testDSizee = sys.getsizeof(testData)
	# trainLSize = sys.getsizeof(trainLabels)
	# validLSize = sys.getsizeof(validationLabels)
	# testLSizee = sys.getsizeof(testLabels)
	# print("traindatsize: ", trainDSize/1000000)
	# print("validatisize: ", validDSize/1000000)
	# print("testdatasize: ", testDSizee/1000000)
	# print("trainLabsize: ", trainLSize/1000000)
	# print("valiLabisize: ", validLSize/1000000)
	# print("testLabasize: ", testLSizee/1000000)
	# print("total: MB", (trainDSize + validDSize + testDSizee + trainLSize + validLSize + testLSizee)/1000000)

	return trainLabels, trainData, \
		   validationLabels, validationData, \
		   testLabels, testData

#Detect AFEW faces and save 3 channel greyscales to .jpg
#in dir structure: datasets/AFEW/<setName>/imgs/(labelFolders)/*faceimages.jpg*
#out dir structure: datasets/AFEWParsed/datasets/AFEW/<setName>/imgs/(labelFolders)
#setName: Train/Val/Test
def afewExtractor(setName):
	paths = os.walk('datasets/AFEW/'+setName+'/imgs')
	cascPath = 'haarcascade_frontalface_default.xml'
	faceCascade = cv2.CascadeClassifier(cascPath)

	for root, dirs, files in paths:
		print('Root:', root)
		print('Dirs:', dirs)
		print('Files:', files)
		count = 0
		for imgName in files:
			if imgName[-4:len(imgName)] == '.jpg':
				img = cv2.imread(root + '/' + imgName)

				# cv2.imshow('window', img)
				# if cv2.waitKey(1) == 27:
				# 	sys.exit(0)

				facesLocs = faceCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), maxSize=(1000, 1000), flags=cv2.CASCADE_SCALE_IMAGE)
				# print('FacesN:', len(facesLocs))
				if count % 200 == 0:
					print('Processing:', count, root, imgName)
				count+=1
				if len(facesLocs) == 1:
					for (x, y, w, h) in facesLocs:
						face = img[y:y + h, x:x + w]
						face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
						face = cv2.resize(face, (48, 48))
						cv2.imwrite('datasets/AFEWParsed/'+root+'/'+imgName, face)

						# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
						# cv2.imshow('window', img)
						# faceBig = cv2.resize(face, (face.shape[0] * 3, face.shape[1] * 3))
						# cv2.imshow('face', faceBig)
						# cv2.waitKey(0)

def parseAFEW(setName):
	paths = os.walk('datasets/AFEWParsed/'+setName)
	labels = []
	data = []
	categories = {'datasets/AFEWParsed/'+setName+'/Angry' : 0,
				  'datasets/AFEWParsed/'+setName+'/Disgust': 1,
				  'datasets/AFEWParsed/'+setName+'/Fear': 2,
				  'datasets/AFEWParsed/'+setName+'/Happy': 3,
				  'datasets/AFEWParsed/'+setName+'/Sad': 4,
				  'datasets/AFEWParsed/'+setName+'/Surprise': 5,
				  'datasets/AFEWParsed/'+setName+'/Neutral': 6}
	shapeTuple = getShapeTuple(48, 1)

	for root, dirs, files in paths:
		print('Root:', root)
		print('Dirs:', dirs)
		# print('Files:', files)
		count = 0
		for imgName in files:
			if count % 500 == 0:
				print('parsing:', count, root)
			count += 1
			labels.append(np.asarray([categories[root]]))
			face = cv2.imread(root + '/' + imgName)
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			face = np.reshape(face, shapeTuple)
			data.append(face)

	labels = to_categorical(labels, num_classes=len(np.unique(labels)))
	data = np.asarray(data)
	print('channelMode: ', image_data_format(), ' data shape:', data.shape)
	data = data.astype('float32')
	data /= 255
	for imgi in range(len(data)):
		data[imgi] = (data[imgi]-0.507566)**2/0.255003 #Normalising to Fer values

	# for i in range(len(data)):
	# 	# if labels[i][6]==1: #show specific emotion example
	# 	if i % 400 == 0:
	# 		print(i, labels[i])
	# 		faceBig = cv2.resize(data[i], (data[i].shape[0] * 3, data[i].shape[1] * 3))
	# 		cv2.imshow('img', faceBig)
	# 		if cv2.waitKey(0) == 27:
	# 			sys.exit(0) # esc to quit

	print('ParsedAFEW - RETURNING UNSHUFFLED')
	return labels, data

def getShapeTuple(rowDim, channels):
	if image_data_format() == 'channels_last':
		shapeTuple = (rowDim, rowDim, channels)
	elif image_data_format() == 'channels_first':
		shapeTuple = (channels, rowDim, rowDim)
	else:
		print("channel not first or last, aborting")
		sys.exit()
	return shapeTuple


if __name__ == '__main__':
	# data = readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))
	afewExtractor('Val')
	# parseAFEW()
