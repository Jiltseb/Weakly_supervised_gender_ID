#!/usr/bin/python3

import numpy
import scipy.io.wavfile as wav
from sphfile import SPHFile
##import ipdb

def wav2feat (wavepath, param=None):
    if param == None:
        param = {'windowLength' : 250,    ## milliseconds
                 'windowShift'  : 10,     ## milliseconds
                 'fs'           : 8000,  ## Sampling rate in Hertz
                 'stdFloor'     : 1e-3 }  ## Floor on standard deviation
        param['windowLengthSamples'] = int(param['windowLength'] * param['fs'] / 1000.0)
        param['windowShiftSamples'] = int(param['windowShift'] * param['fs'] / 1000.0)

    ## Read data and labels
    fs,data = wav.read(wavepath)
    #data = numpy.sum (data1, axis=1)
    fs = 8000
    ## Check number of channels and sampling rate
    assert len(data.shape)==1, 'ERROR: {:s} has multiple channels. Modify the code accordingly and re-run'.format(wavepath)
    assert fs==param['fs'], 'ERROR: Sampling rate mismatch with {:s}: expected {:f}, got {:f}'.format(wavepath, fs, param['fs'])

    ## Remove DC offset
    data = data - numpy.sum(data).astype(numpy.float32)

    ## Normalise amplitude
    norm = numpy.max(numpy.abs(data)).astype(numpy.float32)
    if norm == 0:
        raise ValueError ('{:s} contains all zeros. Discarding'.format(wavepath))
    data = data / norm

    ## Append zeros to data if necessary
    while len(data) < param['windowLengthSamples']:
        ##print ('WARNING: {:s} contains fewer samples than {:d}. Replicating data samples'.format(wavepath, param['windowLengthSamples']))
        ##ipdb.set_trace()
        data = numpy.concatenate([data, data[len(data)-param['windowLengthSamples']:][::-1]])

    ## Determine the number of frames
    numFr = int((len(data)-param['windowLengthSamples'])//param['windowShiftSamples']+1)

    ## Convert Channel-1 of data into a feature matrix
    stride = data.strides[-1]
    feat = numpy.lib.stride_tricks.as_strided(data, shape=(numFr, param['windowLengthSamples']), strides=(param['windowShiftSamples']*stride,stride))

    ## Normalise feature matrix
    std = feat.std(axis=-1)
    std[std < param['stdFloor']] = param['stdFloor']
    feat = ((feat.T - feat.mean(axis=-1))/std).T
           
    return feat

## Load a list of data
def loadListData (wavLabListFile, param=None):
    if param == None:
        param = {'windowLength' : 250,    ## milliseconds
                 'windowShift'  : 10,     ## milliseconds
                 'fs'           : 8000,  ## Sampling rate in Hertz
                 'stdFloor'     : 1e-3 }  ## Floor on standard deviation
        param['windowLengthSamples'] = int(param['windowLength'] * param['fs'] / 1000.0)
        param['windowShiftSamples'] = int(param['windowShift'] * param['fs'] / 1000.0)

    featList = []
    labelList = []
    with open(wavLabListFile) as wll:
        for wl in wll:
            w,l = wl.split()
              
            feat = wav2feat(w, param)
            ## Append to list
            featList.append(feat)
            labelList.append([int(l)]*len(feat))

    ## Concatenate
    featMat = numpy.vstack(featList)
    labelMat = numpy.hstack(labelList).astype(numpy.uint16)

    ## Random indices
    randomInd = numpy.array(range(len(featMat)))
    numpy.random.shuffle(randomInd)

    return featMat[randomInd], labelMat[randomInd]

