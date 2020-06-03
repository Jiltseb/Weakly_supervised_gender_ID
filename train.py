#!/usr/bin/python3

import keras
from rawGenerator import rawGenerator
from optparse import OptionParser
import numpy
import sys
import os
#from sklearn.metrics import recall_score
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
usage = 'USAGE: %prog [options] train.txt cv.txt outdir'

parser = OptionParser(usage=usage)
parser.add_option("-w", "--windowLength", type="int", 
		help="Window length in milliseconds", dest="windowLength", default=250)
parser.add_option("-s", "--windowShift", type="int",
        help="Window shift in milliseconds",  dest="windowShift", default=10)
parser.add_option("-f", "--fs", type="int",
        help="Sampling frequency in Hertz",  dest="fs", default=8000)
opts, args = parser.parse_args()

if len(args) != 3:
    parser.usage += '\n\n' + parser.format_option_help()
    parser.error('Wrong number of arguments')

data_tr = args[0]
data_cv = args[1]
exp		= args[2]

## Feature extraction parameters
param = {'windowLength':opts.windowLength, 'windowShift':opts.windowShift,
         'fs':opts.fs}

## Learning parameters
learning = {'rate' : 0.1,
            'minEpoch' : 4, #was 4
            'lrScale' : 0.5,
            'batchSize' : 256,
            'lrScaleCount' : 6,
            'minValError' : 0.005}#was 005

param['stdFloor'] = 1e-3  ## Floor on standard deviation
param['windowLengthSamples'] = int(param['windowLength'] * param['fs'] / 1000.0)
param['windowShiftSamples'] = int(param['windowShift'] * param['fs'] / 1000.0)

os.makedirs (exp, exist_ok=True)

## Data generators for training and crossvalidation
trGen = rawGenerator (data_tr, learning['batchSize'], param)
cvGen = rawGenerator (data_cv, learning['batchSize'], param)

#class Metrics(Callback):
#    def on_train_begin(self, logs={}):
#     self.val_f1s = []
#     self.val_recalls = []
#     self.val_precisions = []
# 
#    def on_epoch_end(self, epoch, logs={}):
#     val_predict = (numpy.asarray(self.model.predict(self.model.validation_data[0]))).round()
#     val_targ = self.model.validation_data[1]
#     _val_f1 = f1_score(val_targ, val_predict)
#     _val_recall = recall_score(val_targ, val_predict)
#    _val_precision = precision_score(val_targ, val_predict)
#     self.val_f1s.append(_val_f1)
#     self.val_recalls.append(_val_recall)
#     self.val_precisions.append(_val_precision)
#    # print " -val_f1: %f -val_precision: %f -val_recall %f" %(_val_f1, _val_precision, _val_recall)
#     return

## Initialise learning parameters and models
s = keras.optimizers.SGD(lr=learning['rate'], decay=0, momentum=0.5, nesterov=True)

## Model definition
numpy.random.seed(512)
m = keras.models.Sequential([
                    keras.layers.Reshape ((trGen.inputFeatDim, 1), input_shape=(trGen.inputFeatDim,)),
                    keras.layers.Conv1D(filters=80, kernel_size=30,strides=10),#30 
                    keras.layers.pooling.MaxPooling1D(3),
                    keras.layers.Activation('relu'),
                    keras.layers.Conv1D(filters=60, kernel_size=7),
                    keras.layers.pooling.MaxPooling1D(3),
                    keras.layers.Activation('relu'),
                    keras.layers.Conv1D(filters=60, kernel_size=7),
                    keras.layers.pooling.MaxPooling1D(3),
                    keras.layers.Activation('relu'),
                    keras.layers.Flatten(),
                    keras.layers.Dense(100),
                    keras.layers.Activation('relu'),
                    keras.layers.Dense(trGen.outputFeatDim),
                    keras.layers.Activation('softmax')])

## Initial training
m.compile(loss='sparse_categorical_crossentropy', optimizer=s, metrics=['accuracy'])
#metrics = Metrics()
h = [m.fit_generator (trGen, steps_per_epoch=trGen.numSteps,
        validation_data=cvGen, validation_steps=cvGen.numSteps,
        epochs=learning['minEpoch']-1, verbose=2)]
m.save (exp + '/dnn.nnet.h5', overwrite=True)
sys.stdout.flush()
sys.stderr.flush()

valErrorDiff = 1 + learning['minValError'] ## Initialise


## Continue training till validation loss stagnates
while learning['lrScaleCount']:
    print ('Learning rate: %f' % learning['rate'])
    h.append (m.fit_generator (trGen, steps_per_epoch=trGen.numSteps,
            validation_data=cvGen, validation_steps=cvGen.numSteps,
            epochs=1, verbose=2))
    m.save (exp + '/dnn.nnet.h5', overwrite=True)
    sys.stdout.flush()
    sys.stderr.flush()

    ## Check validation error and reduce learning rate if required
    valErrorDiff = h[-2].history['val_loss'][-1] - h[-1].history['val_loss'][-1]
    if valErrorDiff < learning['minValError']:
        learning['rate'] *= learning['lrScale']
        learning['lrScaleCount'] -= 1
        keras.backend.set_value(m.optimizer.lr, learning['rate'])

