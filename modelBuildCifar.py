from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import sys
from WRN import makeModel

#tfCallback = TensorBoard(log_dir='./TB', histogram_freq=0, write_graph=True, write_images=True)

nEpochs = 200
batch_size = 128 #default 128
nClasses = 10
learningRateDecayRatio = 0.2

# The data, shuffled and split between train and test sets:
(trainData, trainLabels), (testData, testLabels) = cifar10.load_data()

print('trainData shape:', trainData.shape)
print(trainData.shape[0], 'train samples')
print(testData.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
trainLabels = to_categorical(trainLabels, nClasses)
testLabels = to_categorical(testLabels, nClasses)

#Normalise Cifar data. #TODO: meanstd, data augmentation
trainData = trainData.astype('float32')
testData = testData.astype('float32')
trainData /= 255
testData /= 255

wrn = 'WRN-28-4'
n = sys.argv[1]
k = sys.argv[2]
model = makeModel(trainData.shape[1:], #input shape (check channels_first/last)
				  nClasses, #number of classes
				  int(n), int(k))

sgdOpt = SGD(lr=0.1, momentum=0.9, decay=0.0005, nesterov=True) #dampening == 0?

model.compile(loss='categorical_crossentropy',
              optimizer=sgdOpt,
              metrics=['accuracy'])

for epochi in range(1, nEpochs+1):
	if epochi % 20 == 0:
		print('EPOCH:', epochi, 'of:', nEpochs)

		print('EVALUATING')
		evalResult = model.evaluate(testData, testLabels)
		print('\n\nmets: ', model.metrics_names)
		print('evalResult: ', evalResult)

		outStrTxt = 'results' + wrn + '-Cifar-E' + str(epochi) + '.txt'
		outStrh5 = 'results' + wrn + '-Cifar-E' + str(epochi) + '.h5'
		with open(outStrTxt, 'w') as f:
			f.write(wrn + '-B(3,3)')
			f.write('\nmets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
			f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))

		model.save(outStrh5)
		print('model saved.')

	if epochi in [60, 120, 160]:
		K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * learningRateDecayRatio)
		print('LR CHANGED:', K.eval(model.optimizer.lr))
	model.fit(trainData, trainLabels,
			  batch_size=batch_size,
			  epochs=1,
			  validation_split=0.15,
			  shuffle=True)
			  #callbacks=[tfCallback])

'''
print('LR INIT:', K.eval(model.optimizer.lr))
model.fit(trainData, trainLabels,
		  batch_size=batch_size,
		  epochs=60, #1->60
		  validation_split=0.15,
		  shuffle=True,
		  callbacks=[tfCallback])
K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * learningRateDecayRatio)
print('LR CHANGED:', K.eval(model.optimizer.lr))
model.fit(trainData, trainLabels,
		  batch_size=batch_size,
		  epochs=60, #61->120
		  validation_split=0.15,
		  shuffle=True)
K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * learningRateDecayRatio)
print('LR CHANGED:', K.eval(model.optimizer.lr))
model.fit(trainData, trainLabels,
		  batch_size=batch_size,
		  epochs=40, #121->160
		  validation_split=0.15,
		  shuffle=True)
K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * learningRateDecayRatio)
print('LR CHANGED:', K.eval(model.optimizer.lr))
model.fit(trainData, trainLabels,
		  batch_size=batch_size,
		  epochs=40, #161->200
		  validation_split=0.15,
		  shuffle=True)

'''

print('EVALUATING')
evalResult = model.evaluate(testData, testLabels)

print('\n\nmets: ', model.metrics_names)
print('evalResult: ', evalResult)
with open('results' + wrn + '-CifarFin.txt', 'w') as f:
	f.write(wrn + '-B(3,3)')
	f.write('mets: ' + model.metrics_names[0] + ' ' + model.metrics_names[1])
	f.write('\nevalResult: ' + str(evalResult[0]) + ' ' + str(evalResult[1]))

model.save(wrn + '-CifarFin.h5')
print('model saved.')

del model