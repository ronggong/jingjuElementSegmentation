# -*- coding: utf-8 -*-

import sys, os
import matplotlib.pyplot as plt
import time

# add pYin src path
pYinPath = '../pypYIN/src'
sys.path.append(pYinPath)

# add src path
dir = os.path.dirname(os.path.realpath(__file__))
srcpath = dir+'/src'
sys.path.append(srcpath)

import pYINmain
import pitchtrackSegByNotes
import essentia.standard as ess
import numpy as np

if __name__ == "__main__":

    start_time = time.time()  # starting time

    '''
    # initialise
    filename1 = '/home/rgong/Music/bestQuality/dan/1-08 伴奏：玉堂春跪在督察院.wav'
    #filename1 = pYinPath + '/testAudioLong.wav'
    samplingFreq = 44100
    frameSize = 2048
    hopSize = 256

    # pYin
    pYinInst = pYINmain.PyinMain()
    pYinInst.initialise(channels = 1, inputSampleRate = samplingFreq, stepSize = hopSize, blockSize = frameSize,
                   lowAmp = 0.80, onsetSensitivity = 0.7, pruneThresh = 0.1)

    # frame-wise calculation
    audio = ess.MonoLoader(filename = filename1, sampleRate = samplingFreq)()
    for frame in ess.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize):
        fs = pYinInst.process(frame)

    # calculate smoothed pitch and mono note
    fs = pYinInst.getRemainingFeatures()

    # output and concatenate smoothed pitch track
    pitchtrack = []
    print 'pitch track'
    for ii in fs.m_oSmoothedPitchTrack:
        pitchtrack.append(ii.values)
        print ii.values
    print '\n'

    # output of mono notes,
    # column 0: frame number,
    # column 1: pitch in midi numuber, this is the decoded pitch
    # column 2: attack 1, stable 2, silence 3
    print 'mono note decoded pitch'
    for ii in fs.m_oMonoNoteOut:
        print ii.frameNumber, ii.pitch, ii.noteState
    print '\n'

    '''

    # segmentation
    # ptSeg = pitchtrackSegByNotes.pitchtrackSegByNotes(samplingFreq, frameSize, hopSize)

    ptSeg = pitchtrackSegByNotes.pitchtrackSegByNotes()

    #ptSeg.doSegmentation(pitchtrack, fs.m_oMonoNoteOut)

    pitchtrack_filename = './pYinOut/nirenxinPitchtrack.txt'
    monoNoteOut_filename = './pYinOut/nirenxinMonoNoteOut.txt'

    ptSeg.doSegmentationForPyinVamp(pitchtrack_filename, monoNoteOut_filename)

    ptSeg.pltNotePitchtrack(saveFig=True, figFolder='../jingjuSegPic/')


    print("--- %s seconds ---" % (time.time() - start_time))





