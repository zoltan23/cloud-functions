import instrument_classification_module
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from scipy.io import wavfile as wav
import numpy as np

from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from keras.models import load_model

def predict(filename, instrument):
    predicted_instrument = instrument_classification_module.classify(filename)
    if (instrument == predicted_instrument):
        print('TEST PASSED!!!!! Expected' + instrument + ' and got ' + predicted_instrument )
    else:
        print('TEST FAILED!!!!! Expected' + instrument + ' and got ' + predicted_instrument )

def predictedTrumpetTest():
    predict('/tf/desktop/Audio-Classification/clean/229be2be.wav', "Trumpet")

def predictedClarinetTest():
    predict('/tf/desktop/Audio-Classification/clean/229be2be.wav', "Trumpet")    


    

# predictedInstrumentTest('/tf/desktop/Audio-Classification/clean/229be2be.wav', "Saxophone")
# predictedInstrumentTest('/tf/desktop/Audio-Classification/clean/229be2be.wav', "Clarinet")
