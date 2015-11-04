# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from utilFunc import pitch2midi


class pitchtrackSegByNotes(object):

    def __init__(self, fs = 44100, frameSize = 2048, hopSize = 256):
        self.fs = fs
        self.frameSize = frameSize
        self.hopSize = hopSize

        self.max = 0
        self.min = 0

        self.reset()

    def reset(self):
        self.noteStartEndFrame = []
        self.pitchtrackByNotes = []

    def noteEndFrameHelper(self, notePitchtrack, startDur):

        notePitchtrack = np.abs(notePitchtrack)
        notePitchtrack = pitch2midi(notePitchtrack)  # convert to midi note

        self.pitchtrackByNotes.append([notePitchtrack,startDur])
        notePitchtrack = []
        startDur = [0, 0]

        return notePitchtrack, startDur

    def minMaxPitchtrack(self, pitchtrack):

        ptPositive = [item for item in pitchtrack if item > 0]
        self.max = max(ptPositive)
        self.min = min(ptPositive)

        return

    def doSegmentation(self, pitchtrack, monoNoteOut):

        '''
        [ [pitchtrack of note 1, [startingFrame1, durationFrame1]],
                   [pitchtrack of note 2, [startingFrame2, durationFrame2]], ... ... ]

        :param pitchtrack: smoothed pitchtrack output from pYin
        :param monoNoteOut: note pitchtrack output from pYin note transcription
        :return: self.pitchtrackByNotes,
        '''

        # get the max and min values of pitch track
        self.minMaxPitchtrack(pitchtrack)

        # initialisation
        jj = 0
        old_ns = 4
        mnoLen = len(monoNoteOut)
        notePitchtrack = []
        startDur = [0, 0]

        for ii in monoNoteOut:
            ns = ii.noteState
            if (jj == 0 or old_ns == 3) and ns == 1:
                #  note attack on first frame or note attack frame
                notePitchtrack.append(pitchtrack[jj][0])
                startDur[0] = ii.frameNumber
                startDur[1] += 1
            if old_ns == 2 and ns == 3:
                #  note end frame
                notePitchtrack, startDur = \
                    self.noteEndFrameHelper(notePitchtrack, startDur)
            if old_ns == 2 and jj == mnoLen-1:
                #  last frame note and track frame
                notePitchtrack, startDur = \
                    self.noteEndFrameHelper(notePitchtrack, startDur)
            if old_ns == 2 and (ns == 2 or ns == 1):
                #  note stable frame
                notePitchtrack.append(pitchtrack[jj][0])
                startDur[1] += 1

            old_ns = ns
            jj += 1

        return

    def readPyinPitchtrack(self, pitchtrack_filename):

        '''
        :param pitchtrack_filename:
        :return: frameStartingTime, pitchtrack
        '''

        pitchtrack = np.loadtxt(pitchtrack_filename)
        frameStartingTime = pitchtrack[:,0]
        pitchtrack = pitchtrack[:,1]

        return frameStartingTime, pitchtrack

    def readPyinMonoNoteOut(self, monoNoteOut_filename):

        '''
        :param monoNoteOut_filename:
        :return: noteStartingTime, noteDurationTime
        '''

        monoNoteOut = np.loadtxt(monoNoteOut_filename)
        noteStartingTime = monoNoteOut[:,0]
        noteDurTime = monoNoteOut[:,2]

        return noteStartingTime, noteDurTime

    def doSegmentationForPyinVamp(self, pitchtrack_filename, monoNoteOut_filename):

        # doSegmentationFunction for pYin vamp plugin exported
        # pitchtrack and monoNote

        self.reset()

        frameStartingTime, pitchtrack = self.readPyinPitchtrack(pitchtrack_filename)
        noteStartingTime, noteDurTime = self.readPyinMonoNoteOut(monoNoteOut_filename)

        self.minMaxPitchtrack(pitchtrack)

        pitchtrack = np.abs(pitchtrack)
        pitchtrack = pitch2midi(pitchtrack)

        noteEndingTime = noteStartingTime+noteDurTime

        noteStartingIndex = []
        noteEndingIndex = []

        for ii in noteStartingTime:
            noteStartingIndex.append(np.argmin(np.abs(frameStartingTime - ii)))

        for ii in noteEndingTime:
            noteEndingIndex.append(np.argmin(np.abs(frameStartingTime - ii)))

        for ii in range(len(noteStartingIndex)):
            notePitchtrack = pitchtrack[noteStartingIndex[ii]:(noteEndingIndex[ii]+1)]
            startDur = [noteStartingIndex[ii],noteEndingIndex[ii]-noteStartingIndex[ii]+1]

            noteStartingFrame = int(noteStartingTime[ii]*(44100/256))
            noteEndingFrame = int(noteEndingTime[ii]*(44100/256))
            self.noteStartEndFrame.append([noteStartingFrame,noteEndingFrame])
            self.pitchtrackByNotes.append([notePitchtrack.tolist(), startDur])

        return

    def pltNotePitchtrack(self, saveFig = False, figFolder = './'):

        '''
        :param notePitchtrack: [pitch1, pitch2, ...]
        :param startDur: [starting frame, duration frame]
        :return:
        '''

        if not os.path.exists(figFolder):
            os.makedirs(figFolder)

        jj = 1
        for ii in self.pitchtrackByNotes:
            notePitchtrack = ii[0]
            startDur = ii[1]

            # time in s
            startingTime = (self.frameSize/2 + self.hopSize*startDur[0])/float(self.fs)
            durTime = (startDur[1]+1)*self.hopSize/float(self.fs)
            frameRange = range(startDur[0], startDur[0]+startDur[1])

            plt.figure()
            plt.plot(frameRange, np.abs(notePitchtrack))
            plt.ylabel('midi note number, 69: A4')
            plt.xlabel('frame')
            plt.title('starting time: ' + str(startingTime) +
                      ' duration: ' + str(durTime))

            axes = plt.gca()
            #axes.set_xlim([xmin,xmax])
            #axes.set_ylim([self.min-5,self.max+5])

            if saveFig == True:
                plt.savefig(figFolder+str(jj)+'.png')
                plt.close()
            jj += 1

        if saveFig == False:
            plt.show()