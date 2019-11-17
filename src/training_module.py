
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

def trainModel():
    filename = '/tf/desktop/Audio-Classification/clean/a64a3740.wav'
    librosa_audio, librosa_sample_rate = librosa.load(filename) 
    scipy_sample_rate, scipy_audio = wav.read(filename) 

    print('Original sample rate:', scipy_sample_rate) 
    print('Librosa sample rate:', librosa_sample_rate) 

    print('Original audio file min~max range:', np.min(scipy_audio), 'to', np.max(scipy_audio))
    print('Librosa audio file min~max range:', np.min(librosa_audio), 'to', np.max(librosa_audio))


    # Original audio with 2 channels 
    plt.figure(figsize=(12, 4))
    plt.plot(scipy_audio)

    plt.figure(figsize=(12, 4))
    plt.plot(librosa_audio)

    mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
    print(mfccs.shape)

    librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')

    def extract_features(file_name):
   
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 

            mask = envelope(audio, sample_rate, 0.0005)

            mfccs = librosa.feature.mfcc(y=audio[mask], sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)

            #print("mfccsscaled", mfccsscaled.shape)
        
        except Exception as e:
            print("Error encountered while parsing file: ", file)
            return None 
     
        return mfccsscaled

    def envelope(y, rate, threshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    fulldatasetpath = '/tf/desktop/Audio-Classification/wavfiles/'

    metadata = pd.read_csv('/tf/desktop/Audio-Classification/instruments.csv')

    features = []

    # Iterate through each sound file and extract the features 
    for index, row in metadata.iterrows():
    
        file_name = os.path.join(os.path.abspath(fulldatasetpath), str(row["fname"]))

        class_label = row["label"]
        data = extract_features(file_name)
        features.append([data, class_label])

        # Convert into a Panda dataframe 
        featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

        #print('Finished feature extraction from ', len(featuresdf), ' files') 


    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # Encode the classification labels
    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    # split the dataset 
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

    num_labels = yy.shape[1]
    filter_size = 2

    model = Sequential()

    model.add(Dense(256, input_shape=(40,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 

    # Display model architecture summary 
    #model.summary()

    # Calculate pre-training accuracy 
    score = model.evaluate(x_test, y_test, verbose=0)
    accuracy = 100*score[1]

    print("Pre-training accuracy: %.4f%%" % accuracy)

    num_epochs = 100
    num_batch_size = 32
    
    #weights.best.basic_mlp
    #checkpointer = ModelCheckpoint(filepath='/tf/desktop/Audio-Classification/saved_models/model1.hdf5', verbose=1, save_best_only=True)
    start = datetime.now()
    
    #callbacks=[checkpointer],
    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),  verbose=1)
    model.save('/tf/desktop/cloud-functions/models/model1.h5')

    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    # Evaluating the model on the training and testing set
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Training Accuracy: ", score[1])

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Testing Accuracy: ", score[1])
    print("training module ran!")

trainModel()
