#!/usr/bin/python3

import numpy
from wavIO import wav2feat
import scipy.io.wavfile as wav
#import pdb

class rawGenerator:
    def __init__ (self, wavLabListFile, batchSize=256, param=None):
        self.wavLabListFile = wavLabListFile
        self.batchSize = batchSize
        self.param = param
        self.maxSplitDataSize = 100

        self.paramDefaults()
        self.computeAttributes()

        numpy.random.seed(512)
        self.wll = open(self.wavLabListFile)
        self.splitDataCounter = 0
   
        self.x = numpy.empty ((0, self.inputFeatDim), dtype=numpy.float32)
        self.y = numpy.empty (0, dtype=numpy.uint16)
        self.batchPointer = 0
        self.doUpdateSplit = True

    def __exit__ (self):
        self.wll.close()

    ## Defaults for feature extraction parameters
    def paramDefaults(self):
        if self.param == None:
            self.param = {'windowLength' : 250,    ## milliseconds
                          'windowShift'  : 10,     ## milliseconds
                          'fs'           : 8000,  ## Sampling rate in Hertz
                          'stdFloor'     : 1e-3 }  ## Floor on standard deviation
            self.param['windowLengthSamples'] = int(self.param['windowLength'] * self.param['fs'] / 1000.0)
            self.param['windowShiftSamples'] = int(self.param['windowShift'] * self.param['fs'] / 1000.0)

    ## Compute attributes for the generator
    def computeAttributes (self):
        self.inputFeatDim = self.param['windowLengthSamples']
        print ('Checking files in {:s}'.format(self.wavLabListFile))
        labels = set()
        with open(self.wavLabListFile) as wll:
            numFr = 0
            numutt = 0
            for wl in wll:
                w,l = wl.split()
                fs, data = wav.read(w)
                #pdb.set_trace()
                #data =numpy.sum (data1, axis=1)
                ## Check number of channels and sampling rate
                assert len(data.shape)==1, 'ERROR: {:s} has multiple channels. Modify the code accordingly and re-run'.format(wavepath)
                assert fs==self.param['fs'], 'ERROR: Sampling frequency mismatch with {:s}: expected {:f}, got {:f}'.format(w, fs, self.param['fs'])
                numFr += max(int((len(data)-self.param['windowLengthSamples'])//self.param['windowShiftSamples']+1),1)
                numutt += 1
                if l not in labels:
                    labels.add(l)
        self.outputFeatDim = len(labels)
        self.numFeats = numFr
        self.numUtterances = numutt
        self.numSplit = -(-self.numUtterances//self.maxSplitDataSize)
        self.numSteps = -(-self.numFeats//self.batchSize)
        print ('Attributes: numFeats={:d}, numUtterances={:d}'.format(self.numFeats, self.numUtterances))

    ## Load another split
    def getNextSplitData (self):
        featList = []
        labelList = []
        for n in range(self.maxSplitDataSize):
            wl = self.wll.readline()
            if not wl:
                break
            w,l = wl.split()
            
            ## Extract features
            feat = wav2feat(w, self.param)
          
            ## Append to list
            featList.append(feat)
            labelList.append([int(l)]*len(feat))

        ## Concatenate
        featMat = numpy.vstack(featList)
        labelMat = numpy.hstack(labelList).astype(numpy.uint16)
        return featMat, labelMat

    ## Make the object iterable
    def __iter__ (self):
        return self

    ## Retrieve a mini batch
    def __next__ (self):
        while (self.batchPointer + self.batchSize >= len (self.x)):
            if not self.doUpdateSplit:
                self.doUpdateSplit = True
                break

            self.splitDataCounter += 1
            x,y = self.getNextSplitData()
            self.x = numpy.concatenate ((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate ((self.y[self.batchPointer:], y))
            self.batchPointer = 0

            ## Shuffle data
            randomInd = numpy.array(range(len(self.x)))
            numpy.random.shuffle(randomInd)
            self.x = self.x[randomInd]
            self.y = self.y[randomInd]

            if self.splitDataCounter == self.numSplit:
                self.wll.seek(0)
                self.splitDataCounter = 0
                self.doUpdateSplit = False
        
        xMini = self.x[self.batchPointer:self.batchPointer+self.batchSize]
        yMini = self.y[self.batchPointer:self.batchPointer+self.batchSize]
        self.batchPointer += self.batchSize
        return (xMini, yMini)
   
