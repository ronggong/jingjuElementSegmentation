# -*- coding: utf-8 -*-

import sys, os
import matplotlib.pyplot as plt
import time
import json
import shutil
import random

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
import noteClass as nc
import featureVecTarget as fvt
import trainTestKNN as ttknn
import refinedSegmentsManipulation as rsm
from vibrato import vibrato

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

    pitchtrackNoteTrainFolderPath = './pYinOut/laosheng/train/'
    pitchContourClassificationModelName = './pYinOut/laosheng/train/model/pitchContourClassificationModel.pkl'
    groundtruthNoteLevelPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/pyinNoteCurvefit/classified'
    groundtruthNoteDetailPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/refinedSegmentCurvefit/classified'
    featureVecTrainFolderPath = './pYinOut/laosheng/train/featureVec/'
    targetTrainFolderPath = './pYinOut/laosheng/train/target/'

    pitchtrackNotePredictFolderPath = './pYinOut/laosheng/predict/'
    featureVecPredictFolderPath = './pYinOut/laosheng/predict/featureVec/'
    targetPredictFolderPath = './pYinOut/laosheng/predict/target/'


    recordingNamesTrain = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']
    recordingNamesPredict = ['weiguojia_section_pro','weiguojia_section_amateur']
    #recordingNamesPredict = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']       # evaluation

    '''
    ############################################## train process #######################################################
    # WARNING!!! don't run train process ANY MORE!!! because the segmentRefinement function is not the same ANY MORE!
    # If this is ran, we need MANUALLY re-prepare the training groundtruch!!! This will take several days!!!
    ######## segmentation and features ########
    nc1 = nc.noteClass()
    nc1.noteSegmentationFeatureExtraction(pitchtrackNoteTrainFolderPath,featureVecTrainFolderPath,recordingNamesTrain,segCoef=0.2)

    ######### construct target json ###########

    fvt1 = fvt.FeatureVecTarget()
    fvt1.constructJson4DetailFeature(featureVecTrainFolderPath,targetTrainFolderPath,recordingNamesTrain,groundtruthNoteDetailPath)

    ################ train ####################
    ttknn1 = ttknn.TrainTestKNN()
    ttknn1.gatherFeatureTarget(featureVecTrainFolderPath,targetTrainFolderPath,recordingNamesTrain)
    # ttknn1.featureVec2DPlot([3,7])
    ttknn1.crossValidation(pitchContourClassificationModelName)
    '''

    ################################################## predict #########################################################

    # segmentation
    nc2 = nc.noteClass()
    nc2.noteSegmentationFeatureExtraction(pitchtrackNotePredictFolderPath,
                                                  featureVecPredictFolderPath,recordingNamesPredict,
                                                  segCoef=0.3137,predict=True)
    # predict
    ttknn2 = ttknn.TrainTestKNN()
    ttknn2.predict(pitchContourClassificationModelName,featureVecPredictFolderPath,
                   targetPredictFolderPath,recordingNamesPredict)

    '''
    ################################################ evaluation code ###################################################
    # uncomment it only when needs evaluation
    with open('./pYinOut/laosheng/predict/evaluationResult02.txt', "w") as outfile:
        for sc in np.linspace(0.2,0.5,30):
            COnOffF,COnF,OBOnRateGT,OBOffRateGT = nc2.noteSegmentationFeatureExtraction(pitchtrackNotePredictFolderPath,
                                                  featureVecPredictFolderPath,recordingNamesPredict,
                                                  segCoef=0.3137,predict=True,evaluation=True)
            outfile.write(str(sc)+'\t'+str([COnOffF,COnF,OBOnRateGT,OBOffRateGT])+'\n')
    '''

    '''

    ########################################### representation #########################################################

    for rm in recordingNamesPredict:
        targetFilename = targetPredictFolderPath+rm+'.json'
        refinedSegmentFeaturesFilename = pitchtrackNotePredictFolderPath+rm+'_refinedSegmentFeatures.json'
        # representationFilename = pitchtrackNotePredictFolderPath+rm+'_representation.txt'
        representationFilename = pitchtrackNotePredictFolderPath+rm+'_representation.json'
        figureFilename = pitchtrackNotePredictFolderPath+rm+'_reprensentationContourFigure.png'
        pitchtrackFilename = pitchtrackNotePredictFolderPath+rm+'_regression_pitchtrack.txt'

        rsm1 = rsm.RefinedSegmentsManipulation()
        rsm1.process(refinedSegmentFeaturesFilename,targetFilename,
                     representationFilename,figureFilename,pitchtrackFilename)
'   '''

    '''
    ################################################## copy code #######################################################
    #  below code to 1) get class note name from classified folder, 2) copy the midinote png with the same name into
    #  classified + 'midinote' folder
    nc1 = ncc.noteClass()

    prependPathClassified = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/classified'
    prependPath = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train'

    recordingNamesClassified = ['male_02_neg_01', 'male_12_neg_01', 'male_12_pos_01', 'male_13_pos_01']
    recordingNames = ['male_02/neg_1_midinote', 'male_12/neg_1_midinote', 'male_12/pos_1_midinote', 'male_13/pos_1_midinote']

    for ii in range(len(recordingNames)):
        ncd = nc1.noteNumberOfClass(prependPathClassified, recordingNamesClassified[ii])

        pathnameMO = os.path.join(prependPath, recordingNames[ii])
        onlypngs = [ f for f in os.listdir(pathnameMO) if f.endswith('.png') ]

        for key in ncd:

            #  path to move into
            pathnameMI = os.path.join(prependPathClassified, key, recordingNamesClassified[ii]+'midinote')

            if not os.path.exists(pathnameMI):
                os.makedirs(pathnameMI)

            for png in onlypngs:
                print png
                if png in ncd[key]:
                    shutil.copyfile(os.path.join(pathnameMO, png), os.path.join(pathnameMI, png))
    '''


    print("--- %s seconds ---" % (time.time() - start_time))





