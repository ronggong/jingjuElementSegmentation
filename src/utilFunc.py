from math import log, ceil
import numpy as np
import essentia.standard as ess

def pitch2midi(pitchtrack):
    #  convert pitch hz to midi note number

    flag = False
    if not isinstance(pitchtrack, np.ndarray):
        pitchtrack = np.array(pitchtrack)
        flag = True

    midi = 12.0 * np.log(pitchtrack/440.0)/np.log(2.0) + 69.0

    if flag:
        midi = midi.tolist()

    return midi

def midi2pitch(miditrack):
    #  convert midi note number to pitch in hz

    flag = False
    if not isinstance(miditrack, np.ndarray):
        miditrack = np.array(miditrack)
        flag = True

    pitchtrack = np.exp((miditrack-69.0)/12.0*np.log(2.0))*440.0

    return pitchtrack

def vibFreq(pitchtrack, sp, hopsize):
    '''
    :param pitchtrack:
    :param sp: samplerate of wave audio
    :param hopsize:
    :return: 3 frequencies of potential vibrato
    '''

    if pitchtrack.dtype != np.float32:
        pitchtrack = pitchtrack.astype(np.float32)

    pitchtrackPad = pitchtrack[:]

    sampleRate = sp/hopsize
    ptlen = len(pitchtrack)
    fftSize=int(pow(2, ceil(log(ptlen)/log(2))))  # next pow of pitchtrack length
    if ptlen<fftSize:
        pitchtrackPad = np.append(pitchtrack, np.zeros(fftSize-ptlen, dtype=np.float32))
    S = ess.Spectrum(size=fftSize)(pitchtrackPad)
    locs, amps= ess.PeakDetection(maxPeaks=3, orderBy='amplitude')(S)
    freqs = locs*(fftSize/2+1)*sampleRate/fftSize

    return freqs[0]

def vibExt(pitchtrack, sp, hopsize, vibRate):

    frameTime = 1.5/vibRate
    sampleRate = sp/hopsize
    frameSize = int(round(frameTime*sampleRate))
    if len(pitchtrack)>frameSize:
        extents = []
        for jj in range(0,len(pitchtrack)-frameSize,int(round(frameSize/2))):
            frame = pitchtrack[jj:jj+frameSize]
            extents.append(max(frame)-min(frame))
        ext = np.mean(extents)
    else:
        ext = max(pitchtrack)-min(pitchtrack)

    return ext

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth