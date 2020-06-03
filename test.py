#!/usr/bin/python3
import os
import sys
import keras
import numpy
from wavIO import wav2feat
from optparse import OptionParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
#import pdb


def test_files (model, test_wav_lab, param):
    m = keras.models.load_model (model)
    dir_name='/idiap/temp/jsebastian/ComParE_Self/rawCnn/framePost'
    frameTrueList = []
    uttTrueList = []
    framePredList = []
    uttPredList = []
    uttPredList2 = []
    with open(test_wav_lab) as wll:
        for wl in wll:
            wavepath,label = wl.split()

            ## Compute features
            feat = wav2feat(wavepath, param)
            #pdb.set_trace()
            ## Compute predictions
           # _,_,_,_,_,_,_,_,temp=wavepath.split("/")
           # name,_=temp.split(".")
            framePosteriors = m.predict(feat)
           # numpy.savetxt(os.path.join(dir_name,name + "." +'txt'),framePosteriors)            
            #frame_chk = m.predict_proba(feat)
            #print(framePosteriors.shape)
            logavr = numpy.average(numpy.log(framePosteriors), axis=0)
            #print(logavr.shape)
            #ipdb.set_trace()
            uttPred2 = numpy.argmax(logavr)
            uttPosterior = numpy.sum(framePosteriors, axis=0)
            framePred = numpy.argmax(framePosteriors, axis=1)
            uttPred = numpy.argmax(uttPosterior)
            
            framePredList.append (framePred)
            uttPredList.append (uttPred)
            uttPredList2.append (uttPred2)
            ## True labels
            label = int(label)
            frameTrueList += [label]*len(feat)
            uttTrueList += [label]

    framePredList = numpy.hstack(framePredList).astype(numpy.int)

    ## Compute precision, recall, F1 score and support
    pf,rf,ff,sf = precision_recall_fscore_support (frameTrueList, framePredList,average='macro')
    pu,ru,fu,su = precision_recall_fscore_support (uttTrueList, uttPredList,average='macro')
    pu2,ru2,fu2,su2 = precision_recall_fscore_support (uttTrueList, uttPredList2,average='macro')
    conf_mat = confusion_matrix (uttTrueList, uttPredList)
    
    ## Print
    numpy.set_printoptions(formatter={'float':'{: 0.3f}'.format, 'int':'{: 6d}'.format})
    print ('')
    print ('')
    print ('Model       :\t', model)
    print ('List        :\t', test_wav_lab)
    print ('Parameters  :\t', param)
    print ('')
    print ('Frame level scores:')
    print ('Precision   :\t', pf)
    print ('Recall      :\t', rf)
    print ('F1 score    :\t', ff)
    print ('Support     :\t', sf)
    print ('')
    print ('Utterance level scores:')
    print ('Precision   :\t', pu)
    print ('Recall      :\t', ru)
    print ('F1 score    :\t', fu)
    print ('Support     :\t', su)
    print ('')
    print ('')
    print ('Utterance level scores:')
    print ('Precision   :\t', pu2)
    print ('Recall      :\t', ru2)
    print ('F1 score    :\t', fu2)
    print ('Support     :\t', su2)
    print ('')
    print ('')
    print ('Utterance conf mat:')
    print(conf_mat)

if __name__ == '__main__':
    usage = 'USAGE: %prog [options] model test.txt'

    parser = OptionParser(usage=usage)
    parser.add_option("-w", "--windowLength", type="int",
            help="Window length in milliseconds", dest="windowLength", default=250)
    parser.add_option("-s", "--windowShift", type="int",
            help="Window shift in milliseconds",  dest="windowShift", default=10)
    parser.add_option("-f", "--fs", type="int",
            help="Sampling frequency in Hertz",  dest="fs", default=8000)
    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.usage += '\n\n' + parser.format_option_help()
        parser.error('Wrong number of arguments')
    
    model           = args[0]
    test_wav_lab    = args[1]

    ## Feature extraction parameters
    param = {'windowLength':opts.windowLength, 'windowShift':opts.windowShift,
            'fs':opts.fs}

    param['stdFloor'] = 1e-3  ## Floor on standard deviation
    param['windowLengthSamples'] = int(param['windowLength'] * param['fs'] / 1000.0)
    param['windowShiftSamples'] = int(param['windowShift'] * param['fs'] / 1000.0)

    ## Test files and print statistics
    test_files (model, test_wav_lab, param)
