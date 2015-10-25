import os
import numpy as np
import matplotlib.pyplot as plt
import utilFunc as uf
import statsmodels.api as smapi
from scipy.signal import argrelextrema


class noteClass(object):

    def __init__(self):
        self.basicNoteClasses = {'glissando':0, 'cubic':1,
                            'parabolic':2, 'vibrato':3, 'other':4}

        #  unClassifiedClass: 5
        self.unClassifiedNoteClasses = ['flat+semiVibrato', 'semiVibrato+flat',
                                        'semiVibrato+flat+semiVibrato', 'nonClassified']
        self.samplerate = 44100
        self.hopsize = 256
        self.resetLocalMinMax()
        self.resetDiffExtrema()
        self.resetSegments()
        self.resetRefinedNotePts()

    def resetLocalMinMax(self):
        self.minimaInd = []
        self.maximaInd = []
        self.ySmooth = []

    def resetDiffExtrema(self):
        self.diffX = []
        self.diffAmp = []
        self.diffFusion = []
        self.extrema = []

    def resetSegments(self):
        self.segments = []
        self.threshold = 0

    def resetRefinedNotePts(self):
        self.refinedNotePts = []

    def noteNumberOfClass(self, prependPath, recordingName):

        '''
        :param prependPath: is the pathname prepend to note class name
        :param recordingPathName: is the pathname of recording
        :return:
        '''
        noteClassDict = {}
        for noteClass in self.basicNoteClasses:
            noteClassPath = os.path.join(prependPath, noteClass, recordingName)
            onlypngs = [ f for f in os.listdir(noteClassPath) if f.endswith('.png') ]
            noteClassDict[noteClass] = onlypngs

        return noteClassDict

    def normalizeNotePt(self, notePt):
        '''
        :param notePt: note pitch contour
        :return: x, x-axis series [0,1]; y, y-axis series [0,1]
        '''

        x = np.linspace(0,1,len(notePt))
        notePtNorm = notePt[:]
        #notePtNorm = notePtNorm-min(notePtNorm)
        #notePtNorm = notePtNorm*2/max(notePtNorm)

        #  remove DC
        notePtNorm = notePtNorm - np.mean(notePtNorm)

        return x, notePtNorm

    def pitchContourFit(self, x, y, deg):

        '''
        :param x: x support [0,1]
        :param y: y support [0,1]
        :return: polynomial coef p, residuals, rank, singular_values, rcond
        '''

        if len(y) <= deg+1:  # don't fit the curve if it's too short!
            return

        p= np.polyfit(x=x, y=y, deg=deg, full=False)
        return p

    def pitchContourLMBySM(self,x,y):

        # use statsmodel do linear regression, 1-d.
        # this implementation can detect outliers easily

        X = smapi.add_constant(x, prepend=False)        #  add intercept
        '''
        mod = smapi.RLM(y, X)
        res = mod.fit()
        r2_wls = smapi.WLS(mod.endog, mod.exog, weights=res.weights).fit().rsquared
        print r2_wls, res.params
        '''

        regression = smapi.OLS(y,X).fit()               #  fit model

        # outliers test method 1
        # test = regression.outlier_test()                              #
        # outliers = [ii for ii,t in enumerate(test) if t[2] < 0.5]     # outliers test

        # outliers test method 2: cook's distance
        influence = regression.get_influence()
        (c, p) = influence.cooks_distance               # cook's distance
        threshold = 4.0/(len(x)-1-1)
        outliers = [ii for ii,t in enumerate(c) if t > threshold]

        outliers.sort()

        # remove outliers
        if len(outliers):
            xr = np.delete(x, outliers)
            yr = np.delete(y, outliers)
            yinterp = np.interp(x, xr, yr)                         #  linear interpolation outliers

            regression = smapi.OLS(yinterp,X).fit()               #  fit model

        return regression.params, regression.rsquared

        # print regression.params
        # print regression.rsquared
        # print 'Outliers: ', list(outliers)

    def polyfitRsquare(self, x, y, p):

        if p is None:
            return

        # r-squared
        pc = np.poly1d(p)
        # fit values, and mean
        yhat = pc(x)                         # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        sserr = np.sum((y-yhat)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y-ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        rsqure = 1-sserr/sstot

        return rsqure

    def polyfitVariance(self,x,y,p):

        # the variance of the fitting

        if p is None:
            return

        pc = np.poly1d(p)
        yhat = pc(x)
        var = np.square(np.sum((y-yhat)**2)/len(y))

        return var

    def vibFreq(self, y, deg):

        if len(y) <= deg+1:  #  don't calculate the vib freq if it's too short
            return

        # calculate the vib frequency
        # y should not be the pitch track without normalizing

        x = np.linspace(0,1,len(y))
        p = self.pitchContourFit(x, y, deg)         #  use higher degrees curve fitting
        pc = np.poly1d(p)
        residuals = y-pc(x)                         #  residuals of curve fitting
        residuals = residuals-np.mean(residuals)    #  DC remove

        freq = uf.vibFreq(residuals, self.samplerate, self.hopsize)

        return freq

    ######################################## minima and maxima treatment ###############################################

    def localMinMax(self, y, box_pts=None):

        #  detect local minima and maxima using the smoothed curve

        self.resetLocalMinMax()

        # smooth the signal
        if box_pts is None:
            n = int(len(y)/10)
            box_pts = 3 if n<3 else n
        if box_pts < len(y):
            self.ySmooth = uf.smooth(y, box_pts)
        half_box_pts = np.ceil(box_pts/2.0)

        if len(self.ySmooth):
            # for local maxima
            self.maximaInd = argrelextrema(self.ySmooth, np.greater)
            # remove the boundary effect of convolve
            self.maximaInd = [mi for mi in self.maximaInd[0] if (mi>half_box_pts and mi<len(y)-half_box_pts)]

            # for local minima
            self.minimaInd = argrelextrema(self.ySmooth, np.less)
            # remove the boundary effect of convolve
            self.minimaInd = [mi for mi in self.minimaInd[0] if (mi>half_box_pts and mi<len(y)-half_box_pts)]

        return box_pts

    def diffExtrema(self, x, y):

        #  the absolute difference of amplitudes of consecutive extremas

        self.resetDiffExtrema()

        if (len(self.minimaInd) == 0 and len(self.maximaInd) == 0) or len(self.ySmooth) == 0:
            print 'No minima and maxima detected!'
            return

        #  extrema indices with beginning and ending indices
        self.extrema = [0] + sorted(self.minimaInd+self.maximaInd) + [len(self.ySmooth)-1]

        for ii in range(1,len(self.extrema)):
            da = abs(self.ySmooth[self.extrema[ii]]-self.ySmooth[self.extrema[ii-1]])
            dx = self.extrema[ii]-self.extrema[ii-1]
            self.diffAmp.append(da)
            self.diffX.append(dx)

        #  normalize diffX to [0,1] interval
        diffX = np.array(self.diffX)
        if len(np.unique(diffX)) != 1:  #  if array only contains one unique element, don't do subtraction
            diffX = diffX-np.min(diffX)
        diffX = diffX/float(np.max(diffX))

        self.diffX = diffX.tolist()

        #  the boundary of difference are treated differently
        self.diffAmp[0] = abs(self.ySmooth[self.extrema[1]]-y[0])
        self.diffAmp[-1] = abs(y[-1]-self.ySmooth[self.extrema[-2]])

        diffFusion = np.array(self.diffAmp)*np.array(self.diffX)
        self.diffFusion = diffFusion.tolist()

    def consecutiveSections(self, indices):

        # find consecutive sections of a list
        # example: [1,2,3,5,6,7], return [[1,2,3],[5,6,7]]

        conSections = []
        lenIn = len(indices)
        if lenIn > 1:
            conSec = [indices[0]]
            for ii in range(1,lenIn):
                if ii == lenIn-1:  #  last index
                    if indices[ii] == indices[ii-1]+1:
                        conSec.append(indices[ii])
                        conSections.append(conSec)
                    else:
                        conSections.append(conSec)
                        conSections.append([indices[ii]])
                elif indices[ii] == indices[ii-1]+1:
                    conSec.append(indices[ii])
                else:
                    conSections.append(conSec)
                    conSec = [indices[ii]]
        elif lenIn == 1:
            conSections.append(indices)

        return conSections

    def segmentPointDetection1(self, threshold):

        #  detection of the segmentation point by setting a general y-axis threshold

        # threshold /= len(diffFusion)
        #print threshold

        lenDiffAmp = len(self.diffAmp)
        self.resetSegments()

        if not lenDiffAmp:
            return

        if lenDiffAmp <= 3:  #  we don't consider if there is only one or two extremas case
            return

        cumulDiff = [self.diffAmp[0]]
        for ii in range(1,lenDiffAmp):
            cumulDiff.append(self.diffAmp[ii])
            if np.std(cumulDiff) > threshold:
                self.segments.append(ii)
                cumulDiff = []

    def segmentPointDetection2(self):

        #  detection of the segmentation point by setting a general y-axis threshold

        lenDiffAmp = len(self.diffAmp)
        self.resetSegments()

        if not lenDiffAmp:
            return

        thresholds = []
        if lenDiffAmp > 3:  #  we don't consider if there is only one or two extremas case
            sortedDiffAmp = sorted(self.diffAmp)
            thresholds = sortedDiffAmp[:]

        smallestSdConSecs = [range(0,lenDiffAmp-1)]
        if len(thresholds):
            smallestSd = float("inf")
            for th in thresholds:
                lessThIndices = [n for n,i in enumerate(self.diffAmp) if i<th ]
                largerThIndices = [n for n,i in enumerate(self.diffAmp) if i>=th ]

                # find consecutive sections
                letiConSecs = self.consecutiveSections(lessThIndices[:])
                latiConSecs = self.consecutiveSections(largerThIndices[:])
                conSecs = letiConSecs+latiConSecs

                # check if singleton in section
                singleton = False
                for cs in conSecs:
                    if len(cs) == 1 and cs[0] != 0 and cs[0] != lenDiffAmp-1:
                        singleton = True
                        break

                if not singleton:
                    # mean std
                    sds = []
                    for cs in conSecs:
                        diffAmp = np.array(self.diffAmp)
                        sd = np.std(diffAmp[cs])
                        sds.append(sd)
                    std = np.mean(sds)

                    # smallest std consective sections
                    if std<smallestSd:
                        smallestSd = std
                        smallestSdConSecs = conSecs
                        self.threshold = th

        # segments of extrema index
        for cs in smallestSdConSecs:
            seg = cs[:]
            # seg.append(seg[-1]+1)
            self.segments.append(seg[0])
        self.segments = sorted(self.segments)
        if len(self.segments) > 1:
            self.segments = self.segments[1:]

    def segmentRefinement(self, notePt):

        # construct the refined segmentation pitch contours

        self.resetRefinedNotePts()

        if len(self.segments):
            extrema = np.array(self.extrema)
            segments = extrema[self.segments]
            segments = np.insert(segments, 0, 0)
            segments = np.append(segments, len(notePt)-1)
            for ii in range(1,len(segments)):
                pt = notePt[segments[ii-1]:segments[ii]]
                self.refinedNotePts.append(pt)


    def pltNotePtFc(self, x, y, p, rsquare, vibFreq, saveFig=False, figFolder='./', figNumber=0):

        '''
        plot the pitchtrack and the fitting curve
        '''

        if not os.path.exists(figFolder):
            os.makedirs(figFolder)

        #plt.figure()  #  create a new figure
        ######  pitchtrack figure
        f, ax = plt.subplots()
        ax.plot(x, y, 'b.', label='note pitchtrack')
        if len(self.ySmooth):
            print x, self.ySmooth
            ax.plot(x, self.ySmooth, 'b--', label='smoothed pitchtrack')

        if p is not None:
            pc = np.poly1d(p) # polynomial class
            ax.plot(x, pc(x), 'r-', label='fitting curve')

        '''
        #  draw vibrato part
        if len(vibB) >= 2:
            # sampleRate = 44100/256
            # step = 13

            for ii in range(len(vibB)/2):
                st = vibB[ii*2]
                end = vibB[ii*2+1]
                ax.plot(x[st:end], y[st:end], 'r.')
                #  ax.annotate(str(vibBFreq[0][ii*step:ii*step+1]), xy = (x[st], 0),
                #        xytext=(x[st], 0.05))
        '''

        if len(self.minimaInd) or len(self.maximaInd):
            ax.plot(x[self.minimaInd], self.ySmooth[self.minimaInd], 'gv', markersize=10.0)  #  markers of local minimas
            ax.plot(x[self.maximaInd], self.ySmooth[self.maximaInd], 'g^', markersize=10.0)  #  markers of local maximas

            #  add text above the extreme
            sortedIndices = sorted(self.minimaInd+self.maximaInd)
            for ii in range(len(sortedIndices)):
                ax.annotate(str(ii), xy = (x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]),
                            xytext=(x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]+0.05))

            if len(self.segments):
                for seg in self.segments:
                    ax.axvline(x[self.extrema[seg]], linestyle='--')

        # axarr[0].legend(loc='best')
        ax.set_ylabel('normed midinote number')
        # plt.xlabel('frame')
        ax.set_title('rsquare: '+str(rsquare)+'vibratoFreq: '+str(vibFreq))

        if saveFig==True:
            plt.savefig(figFolder+str(figNumber)+'.png')
            plt.close(f)  #  close the subplot close(f), plt.close() is to close plt

        #####  difference figures
        if len(self.diffX):
            f, axarr = plt.subplots(3, sharex=False)  #  create subplots
            axarr[0].plot(self.diffX)
            axarr[0].set_ylabel('diffs extrema x indice')

            axarr[1].plot(self.diffAmp)
            axarr[1].set_ylabel('diffs extrema amp')
            if self.threshold:
                axarr[1].axhline(self.threshold, linestyle='--')

            axarr[2].plot(self.diffFusion)
            axarr[2].set_ylabel('diffs fusion')


            if saveFig==True:
                plt.savefig(figFolder+'diff_'+str(figNumber)+'.png')
                plt.close(f)  #  close the subplot close(f), plt.close() is to close plt

        if saveFig==False:
            plt.show()


    def pltRefinedNotePtFc(self, x, y, p, rsquare, vibFreq, saveFig=False, figFolder='./', figNumber=0):

        '''
        plot the pitchtrack and the fitting curve
        '''

        if not os.path.exists(figFolder):
            os.makedirs(figFolder)

        #plt.figure()  #  create a new figure
        ######  pitchtrack figure
        f, ax = plt.subplots()
        ax.plot(x, y, 'b.', label='note pitchtrack')
        if len(self.ySmooth) > 3:           # do not print the smooth track if length of ySmooth is too small
            ax.plot(x, self.ySmooth, 'b--', label='smoothed pitchtrack')

        if p is not None:
            pc = np.poly1d(p) # polynomial class
            ax.plot(x, pc(x), 'r-', label='fitting curve')

        if len(self.minimaInd) or len(self.maximaInd):
            ax.plot(x[self.minimaInd], self.ySmooth[self.minimaInd], 'gv', markersize=10.0)  #  markers of local minimas
            ax.plot(x[self.maximaInd], self.ySmooth[self.maximaInd], 'g^', markersize=10.0)  #  markers of local maximas

            #  add text above the extreme
            sortedIndices = sorted(self.minimaInd+self.maximaInd)
            for ii in range(len(sortedIndices)):
                ax.annotate(str(ii), xy = (x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]),
                            xytext=(x[sortedIndices[ii]], self.ySmooth[sortedIndices[ii]]+0.05))

        # axarr[0].legend(loc='best')
        ax.set_ylabel('normed midinote number')
        # plt.xlabel('frame')
        ax.set_title('rsquare: '+str(rsquare)+'vibratoFreq: '+str(vibFreq))

        if saveFig==True:
            plt.savefig(figFolder+str(figNumber)+'.png')
            plt.close(f)  #  close the subplot close(f), plt.close() is to close plt

        if saveFig==False:
            plt.show()
