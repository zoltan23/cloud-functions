import librosa
from scipy.io import wavfile as wav
import numpy as np

filename = '/tf/desktop/Audio-Classification/clean/229be2be.wav' 

def read_wavfile(filename):
    librosa_audio, librosa_sample_rate = librosa.load(filename) 
    scipy_sample_rate, scipy_audio = wav.read(filename) 

    print('Original sample rate:', scipy_sample_rate) 
    print('Librosa sample rate:', librosa_sample_rate)
    return librosa_audio, scipy_audio, librosa_sample_rate, scipy_sample_rate

def wavfile_min_max(scipy_audio, librosa_audio):
    print('Original audio file min~max range:', np.min(scipy_audio), 'to', np.max(scipy_audio))
    print('Librosa audio file min~max range:', np.min(librosa_audio), 'to', np.max(librosa_audio))

def plot_wavfile(scipy_audio):
    import matplotlib.pyplot as plt

    # Original audio with 2 channels 
    plt.figure(figsize=(12, 4))
    plt.plot(scipy_audio)
    print("plot function called")

def plot_merged(librosa_audio):
    import matplotlib.pyplot as plt
    # Librosa audio with channels merged 
    plt.figure(figsize=(12, 4))
    plt.plot(librosa_audio)

def mffcs(librosa_audio, librosa_sample_rate):
    mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
    print(mfccs.shape)