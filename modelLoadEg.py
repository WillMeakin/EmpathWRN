from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import cifar10
from dataParser import readCSV
import sys, cv2
import numpy as np
np.set_printoptions(linewidth=3000)


def ferEg():

	(trainLabels, trainData, validationLabels, validationData, testLabels, testData) = \
		readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))

	img = testData[0:1]
	print('shapeFer:', img[0].shape)
	for i in range(len(trainData)):
		#cv2.imshow('FerImgChannelsFirst', np.reshape(img[0], img[0].shape[::-1]))
		cv2.imshow('FerImgChannelsLast', trainData[i])
		print(trainLabels[i])

		# if testLabels[i][4]==1: #show specific emotion example
		# if i in [0, 1, 2, 6458, 7629, 10423, 11286, 13148, 13402, 13988, 15894, 22198, 22927, 28601]: #black fer images
		# 	print(i, trainLabels[i])
		# 	cv2.imshow('ferImg', trainData[i])
		if cv2.waitKey(0) == 27:
			sys.exit(0) # esc to quit

	#model = load_model('ferCNNChannelsFirst.h5')
	model = load_model('ferCNNChannelsLast.h5')

	evalResult = model.evaluate(testData, testLabels)

	print('\nmets: ', model.metrics_names)
	print('evalResult: ', evalResult)

	print(testLabels[0:1])
	print(model.predict(img, 1, 1))

	del model

def cifarEg():

	nClasses = 10

	# The data, shuffled and split between train and test sets:
	(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()

	print('trainData shape:', trainData.shape)
	print(trainData.shape[0], 'train samples')
	print(testData.shape[0], 'test samples')

	# Convert class vectors to binary class matrices.
	trainLabels = to_categorical(trainLabels, nClasses)
	testLabels = to_categorical(testLabels, nClasses)

	trainData = trainData.astype('float32')
	testData = testData.astype('float32')
	trainData /= 255
	testData /= 255

	model = load_model('WRNCifarTEST.h5')

	evalResult = model.evaluate(testData, testLabels)
	print('\nmets: ', model.metrics_names)
	print('evalResult: ', evalResult)

	img = testData[0:1]

	print(testLabels[0:1])
	pred = model.predict(img, 1, 1)
	print('PREDICITON:\n', pred)

	del model

def classifyFaceCamFer(modelName='WRN-28-4-Fer-Fin.h5'):

	model = load_model(modelName)

	cam = cv2.VideoCapture(1) #TODO: auto get correct cam
	keepRunning = True
	while keepRunning:
		ret_val, img = cam.read()
		grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cascPath = 'haarcascade_frontalface_default.xml'
		faceCascade = cv2.CascadeClassifier(cascPath)
		facesLocs = faceCascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
		prediction = [[-1, -1, -1, -1, -1, -1]]
		faceN = 0
		for (x, y, w, h) in facesLocs:
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			face = grey[y:y+h, x:x+w] #TODO: only does last face detected. Problem?. ALSO: img=colour, grey=greyscale
			face = cv2.resize(face, (48, 48))
			face = face.astype('float32')
			face /= 255
			if np.std(face) != 0:
				face = (face-np.mean(face))**2/np.std(face)
				faceBig = cv2.resize(face, (face.shape[0]*3, face.shape[1]*3))
				cv2.imshow('Face'+str(faceN), faceBig)
				face = np.reshape(face, (1, face.shape[0], face.shape[1], 1)) #reshape to 4d tensor
				#print('shapeCam:', face.shape)
				prediction = model.predict(face, 1, 1)
			else:
				print('Err: std == 0')

			classification = 'UNINITIALISED'
			if max(prediction[0]) == -1:
				classification = 'UNINITIALISED'
			elif prediction[0].argmax() == 0:
				classification = 'ANGRY'
			elif prediction[0].argmax() == 1:
				classification = 'DISGUST'
			elif prediction[0].argmax() == 2:
				classification = 'FEAR'
			elif prediction[0].argmax() == 3:
				classification = 'HAPPY'
			elif prediction[0].argmax() == 4:
				classification = 'SAD'
			elif prediction[0].argmax() == 5:
				classification = 'SUPRISE'
			elif prediction[0].argmax() == 6:
				classification = 'NEUTRAL'
			print('FaceN:', faceN, prediction, classification)
			faceN += 1
			cv2.putText(img, classification, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, 0xffff00)

		cv2.imshow('mywebcam', img)

		if cv2.waitKey(40) == 27:
			keepRunning = False  # esc to quit
	cv2.destroyAllWindows()

	del model

# ferEg()
# cifarEg()
classifyFaceCamFer(modelName=sys.argv[1])