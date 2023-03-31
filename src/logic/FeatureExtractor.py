import numpy as np
import librosa
import scipy
from python_speech_features import mfcc
class FeaturesExtractor:
    def __init__(self):
        pass


    def describe_mfcc(self, wav_mfcc):
        mean = np.mean(wav_mfcc)
        std = np.std(wav_mfcc) 
        maxv = np.amax(wav_mfcc) 
        minv = np.amin(wav_mfcc) 
        median = np.median(wav_mfcc)
        q1 = np.quantile(wav_mfcc, 0.25)
        q3 = np.quantile(wav_mfcc, 0.75)
        iqr = scipy.stats.iqr(wav_mfcc)

        result =  [mean, std, maxv, minv, median, q1, q3, iqr]

        return result

    def extract_features(self, path):
        y, sr = librosa.load(path)
        mfcc_feature = librosa.feature.mfcc(# The audio signal from which to compute features.
                            y=y,
                            # The samplerate of the signal we are working with.
                            sr=sr)
        stats = self.describe_mfcc(mfcc_feature)
        return stats
    
    def extract_raw_features(self, path):
        y, sr = librosa.load(path)
        mfcc_feature = mfcc(
                            y,
                            sr,
                            winlen       = 0.05,
                            winstep      = 0.01,
                            numcep       = 13,
                            nfilt        = 30,
                            nfft         = 1200,
                            appendEnergy = True)
    
        return mfcc_feature
