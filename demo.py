# -*- coding: utf-8 -*-

import sys, os
import matplotlib.pyplot as plt
import time
import json
import shutil
import random
import operator
from sklearn import neighbors

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

    ############################################## segmentation ########################################################

    # ptSeg = pitchtrackSegByNotes.pitchtrackSegByNotes(samplingFreq, frameSize, hopSize)

    ptSeg = pitchtrackSegByNotes.pitchtrackSegByNotes()

    #ptSeg.doSegmentation(pitchtrack, fs.m_oMonoNoteOut)

    nc1 = nc.noteClass()

    recordingNames = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']
    # recordingNames = ['male_02_neg_1']
    for rn in recordingNames:
        pitchtrack_filename = './pYinOut/laosheng/train/'+rn+'_pitchtrack.txt'
        monoNoteOut_filename = './pYinOut/laosheng/train/'+rn+'_monoNoteOut.txt'

        ptSeg.doSegmentationForPyinVamp(pitchtrack_filename, monoNoteOut_filename)

        # ptSeg.pltNotePitchtrack(saveFig=True, figFolder='../jingjuSegPic/laosheng/train/male_13/pos_3_midinote/')

    ###################### calculate the polynomial fitting coefs and vibrato frequency ################################
    #  use pitch track ptseg.pitchtrackByNotes from last step

        featureDict = {}
        segmentsExport = []
        smoothBoxPts = []
        curvefittingDeg = 1
        jj = 1
        for ii in range(len(ptSeg.pitchtrackByNotes)):
            pt = ptSeg.pitchtrackByNotes[ii][0]
            pt = np.array(pt, dtype=np.float32)
            x, y = nc1.normalizeNotePt(pt)                  #  normalise x to [0,1], remove y DC
            sbp = nc1.localMinMax(y)                        #  local minima and extrema of pitch track
            nc1.diffExtrema(x,y)                            #  the amplitude difference of minima and extrema
            nc1.segmentPointDetection1(0.20)

            # # construct the segments frame vector
            # noteStartFrame = ptSeg.noteStartEndFrame[ii][0]
            # noteEndFrame = ptSeg.noteStartEndFrame[ii][1]
            # extremaInd = np.array(nc1.extrema)
            # segmentsInd = extremaInd[nc1.segments]+noteStartFrame
            # segmentsInd = np.insert(segmentsInd,0,noteStartFrame)
            # segmentsInd = np.append(segmentsInd,noteEndFrame)+2     # +2 for sonicVisualizer alignment
            # for sgi in segmentsInd:
            #     segmentsExport.append([sgi,random.random()*len(segmentsInd)])

            #  nc1.segmentPointDetection2()  # segmentation point
            nc1.segmentRefinement(pt)                                               # do the refined segmentation
            #print nc1.refinedNotePts

            for rpt in nc1.refinedNotePts:
                print jj

                xRpt, yRpt = nc1.normalizeNotePt(rpt)                               #  normalise x to [0,1], remove y DC
                nc1.localMinMax(yRpt,sbp)                                           #  local minima and extrema of pitch track
                                                                                    #  use the same smooth pts as before
                # p = nc1.pitchContourFit(xRpt, yRpt, curvefittingDeg)              #  curve fitting
                # rsquare = nc1.polyfitRsquare(xRpt, yRpt, p)                        #  polynomial fitting coefs

                p, rsquare = nc1.pitchContourLMBySM(xRpt,yRpt)                      #  1 degree curve fitting statsmodels
                vibFreq = nc1.vibFreq(rpt, curvefittingDeg)                         #  vibrato frequence

                # featureVec = np.append(p,[rsquare,vibFreq])
                featureVec = np.append([],[rsquare])  #  test different feature vector
                featureDict[jj] = featureVec.tolist()

                #  this plot step is slow, if we only want the features, we can comment this line
                nc1.pltRefinedNotePtFc(xRpt, yRpt, p, rsquare, vibFreq, saveFig=True,
                                        figFolder='../jingjuSegPic/laosheng/train/'+rn+'_curvefit_refined/', figNumber = jj)
                jj += 1

        featureFilename = './pYinOut/laosheng/train/featureVec/'+rn+'.json'
        with open(featureFilename, 'w') as outfile:
            json.dump(featureDict, outfile)

        # with open('./pYinOut/laosheng/train/segment_'+rn+'.txt', "w") as outfile:
        #     for se in segmentsExport:
        #         outfile.write(str(int(se[0]))+'\t'+str(se[1])+'\n')

    ########################################## feature vectors classify ################################################

    recordingNames = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']
    prependPathClassified = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/classified'

    nc1 = nc.noteClass()
    allNoteClasses = list(nc1.basicNoteClasses.keys()) + nc1.unClassifiedNoteClasses

    for rn in recordingNames:
        featureFilename = './pYinOut/laosheng/train/featureVec/'+rn+'.json'
        targetFilename = './pYinOut/laosheng/train/target/'+rn+'.json'
        with open(featureFilename) as data_file:
            featureDict = json.load(data_file)

        targetDict = {}
        for ii in range(1, len(featureDict)+1):
            for noteClass in allNoteClasses:
                noteClassRecordingFoldername = os.path.join(prependPathClassified, noteClass, rn+'midinote')
                if os.path.isdir(noteClassRecordingFoldername):
                    onlypngs = [ f for f in os.listdir(noteClassRecordingFoldername) if f.endswith('.png') ]
                    for png in onlypngs:
                        pngNum = os.path.splitext(png)[0]
                        if str(ii) == pngNum:
                            if noteClass in nc1.basicNoteClasses:
                                #  targetDict[ii] = nc1.basicNoteClasses[noteClass]  #  detail classes
                                targetDict[ii] = 0  # 5 basic classes
                            else:
                                #  targetDict[ii] = 5  #  detail classes
                                targetDict[ii] = 1  # non classified classes

        with open(targetFilename, 'w') as outfile:
            json.dump(targetDict, outfile)



    ################################################## train test ######################################################

    recordingNames = ['male_02_neg_1', 'male_12_neg_1', 'male_12_pos_1', 'male_13_pos_1', 'male_13_pos_3']
    prependPathClassified = '/Users/gong/Documents/pycharmProjects/jingjuSegPic/laosheng/train/classified'
    #  classes = [0,1,2,3,4,5]  #  detail classes
    classes = [0,1]  #  basic and non classified classes

    # collect the feature and target from individual file to one file
    featureDictAll = {}
    targetDictAll = {}
    for rn in recordingNames:
        featureFilename = './pYinOut/laosheng/train/featureVec/'+rn+'.json'
        targetFilename = './pYinOut/laosheng/train/target/'+rn+'.json'

        with open(featureFilename) as data_file:
            featureDict = json.load(data_file)
        with open(targetFilename) as data_file:
            targetDict = json.load(data_file)

        featureKeyList = list(featureDict.keys())
        targetKeyList = list(targetDict.keys())

        for jj in targetKeyList:
            featureDictAll[rn+'_'+jj] = featureDict[jj]
            targetDictAll[rn+'_'+jj] = targetDict[jj]

    shuffledKeys = list(targetDictAll.keys())  #  get the keys which are the recording identifiers
    random.shuffle(shuffledKeys)  #  shuffle the keys

    # partition
    step = len(shuffledKeys)/5

    knn = neighbors.KNeighborsClassifier(n_neighbors=5)

    misclassifiedRateVec = []

    for ii in range(5):
        if ii == 0:
            testKeys = shuffledKeys[ii:step]
        if ii == 4:
            testKeys = shuffledKeys[ii*step:]
        else:
            testKeys = shuffledKeys[ii*step:(ii+1)*step]

        trainKeys = shuffledKeys[:]
        for key2rm in testKeys:
            if key2rm in trainKeys:
                trainKeys.remove(key2rm)

        trainFeatureVec = []
        testFeatureVec =[]
        trainTargetVec =[]
        testTargetVec = []

        for trk in trainKeys:
            trainFeatureVec.append(featureDictAll[trk])
            trainTargetVec.append(targetDictAll[trk])
        for tek in testKeys:
            testFeatureVec.append(featureDictAll[tek])
            testTargetVec.append(targetDictAll[tek])

        # train and test
        knn.fit(trainFeatureVec, trainTargetVec)
        testPredictVec = knn.predict(testFeatureVec)

        # result is a vector which non zero elements are misclassfied
        result = np.array(testTargetVec) - np.array(testPredictVec)

        # result dictionary contains misclassied key:[target class, classified class]
        resultDict = {}
        for rs in range(len(result)):
            if result[rs]:
                resultDict[testKeys[rs]] = [testTargetVec[rs],testPredictVec[rs]]

        # statistics or misclassfied
        resultStatDict = {}
        for tg in classes:
            classesCopy = classes[:]
            classesCopy.remove(tg)
            for cl in classesCopy:
                mcc = str(tg)+str(cl)  #  misclassified combination
                resultStatDict[mcc] = 0
                for rd in resultDict.values():
                    if rd == [tg, cl]:
                        resultStatDict[mcc] += 1

        for rsd in resultStatDict.keys():
            resultStatDict[rsd] /= float(len(resultDict))
        #  print resultStatDict

        #  sort this dictionary by value
        sortedDict = sorted(resultStatDict.items(), key=operator.itemgetter(1), reverse=True)
        # print sortedDict

        misclassified1to0 = 0
        for jj in resultDict.values():
            if jj[0] == 1 and jj[1] == 0:
                misclassified1to0 += 1

        #  misclassifiedRate = len(np.nonzero(result)[0])/float(len(result))
        misclassifiedRate = misclassified1to0/float(len(np.nonzero(result)[0]))
        misclassifiedRateVec.append(misclassifiedRate)
        print misclassifiedRate
        #  print resultStatDict

    print 'mean misclassified rate: '+ str(np.mean(misclassifiedRateVec))

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





