from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
import sys
import numpy as np
from WRN import makeModel
from dataParser import readCSV, parseAFEW

#Run: python3 modelbuild.py <n> <k>
#inputs:
# n = (total number of convs - 4) / 6. See paper.
# k = widening factor
def buildAndTrain(n, k):
	# Tensorboard
	#tfCallback = TensorBoard(log_dir='./TB', histogram_freq=0, write_graph=True, write_images=True)

	#Parse dataset
	(trainLabels, trainData, validationLabels, validationData, testLabels, testData) = \
		readCSV('datasets/fer2013.csv', 48, 1, ('Training', 'PrivateTest', 'PublicTest'))

	# trainLabels, trainData = parseAFEW('Train') #Use AFEW training data
	# validationLabels, validationData = parseAFEW('Val') #Use AFEW training data

	wrn = 'WRN-'+str(int(n)*6+4)+'-'+k #For file naming.
	print(wrn)
	nEpochs = 200
	batch_size = 128 #default 128
	nClasses = 7
	learningRateDecayRatio = 0.2

	#To build a new model
	# model = makeModel(trainData.shape[1:], #input shape (check channels_first/last)
	# 				  nClasses, #number of classes
	# 				  int(n), int(k))
	# sgdOpt = SGD(lr=0.02, momentum=0.9, decay=0.0005, nesterov=True) #dampening == 0?
	# model.compile(loss='categorical_crossentropy',
	# 			 optimizer=sgdOpt,
	# 			 metrics=['accuracy'])

	#To continue training a presaved model
	model = load_model('resultsFer/Best/resultsWRN-28-1-Fer-E45.h5')
	for epochi in range(45, nEpochs+1):
		if epochi % 5 == 0: #checkpoint: evaluate and save model
			print('EPOCH:', epochi, 'of:', nEpochs)

			print('EVALUATING')
			evalResult = model.evaluate(testData, testLabels)
			print('\n\nmets: ', model.metrics_names)
			print('evalResult: ', evalResult)

			outStrTxt = 'results' + wrn + '-E' + str(epochi) + '.txt'
			outStrh5 = 'results' + wrn + '-E' + str(epochi) + '.h5'
			with open(outStrTxt, 'w') as f:
				f.write(wrn + '-B(3,3)')
				f.write('\nmets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
				f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))

			model.save(outStrh5)
			print('model saved.')

		if epochi in [45]: #Change learning rate at these epochs
			K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * learningRateDecayRatio)
			print('LR CHANGED:', K.eval(model.optimizer.lr))

		#Train model
		model.fit(trainData, trainLabels,
				  batch_size=batch_size,
				  epochs=1,
				  validation_data=(validationData, validationLabels),
				  shuffle=True)
				  #callbacks=[tfCallback]) #Tensorboard

	print('EVALUATING')
	evalResult = model.evaluate(testData, testLabels)

	print('\n\nmets: ', model.metrics_names)
	print('evalResult: ', evalResult)
	with open('results' + wrn + '-E' + str(epochi) + '.txt', 'w') as f:
		f.write('mets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
		f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))

	model.save('results' + wrn + '-E' + str(epochi) + '.h5')
	print('model saved.')

	del model

if __name__ == '__main__':
	buildAndTrain(sys.argv[1], sys.argv[2])
