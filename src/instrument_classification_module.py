from scipy.io import wavfile as wav
import numpy as np
import librosa 
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa.display
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import soundfile as sf

class InstrumentClassification:
    def __init__(self):
        filename = '../wavfiles/229be2be.wav' 

        librosa_audio, librosa_sample_rate = librosa.load(filename) 
        scipy_sample_rate, scipy_audio = wav.read(filename) 

        print('Original sample rate:', scipy_sample_rate) 
        print('Librosa sample rate:', librosa_sample_rate) 

        print('Original audio file min~max range:', np.min(scipy_audio), 'to', np.max(scipy_audio))
        print('Librosa audio file min~max range:', np.min(librosa_audio), 'to', np.max(librosa_audio))

        # Original audio with 2 channels 
        plt.figure(figsize=(12, 4))
        plt.plot(scipy_audio)

        # Librosa audio with channels merged 
        plt.figure(figsize=(12, 4))
        plt.plot(librosa_audio)

        mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mfcc=40)
        print(mfccs.shape)

        librosa.display.specshow(mfccs, sr=librosa_sample_rate, x_axis='time')

        # Set the path to the full UrbanSound dataset 
        fulldatasetpath = '../wavfiles'

        metadata = pd.read_csv('../instruments.csv')
        self.instrument_types = metadata.groupby('label').count().to_json()
        print("instrument_types", self.instrument_types)
        features = []

        # Iterate through each sound file and extract the features 
        for index, row in metadata.iterrows():
            
            file_name = os.path.join(os.path.abspath(fulldatasetpath), str(row["fname"]))
            print()
            class_label = row["label"]
            data = self.extract_features(file_name)
            print('file_name', file_name)
            features.append([data, class_label])

        # Convert into a Panda dataframe 
        featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

        print('Finished feature extraction from ', len(featuresdf), ' files') 

        # Convert features and corresponding classification labels into numpy arrays
        X = np.array(featuresdf.feature.tolist())
        y = np.array(featuresdf.class_label.tolist())

        # Encode the classification labels
        le = LabelEncoder()
        yy = to_categorical(le.fit_transform(y)) 
        # split the dataset 
        
        x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

        # import numpy as np
        

        # fix random seed for reproducibility
        # numpy.random.seed(7)

        num_labels = yy.shape[1]
        filter_size = 2

        # Construct model 
        # model = Sequential()

        # model.add(Dense(256, input_shape=(40,)))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        # model.add(Dense(256))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        # model.add(Dense(num_labels))
        # model.add(Activation('softmax'))

        # Tim, here is the new model I came up with.  
        model = Sequential()

        model.add(Dense(256, input_shape=(40,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(11, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 
        # Display model architecture summary 
        model.summary()

        # Calculate pre-training accuracy 
        score = model.evaluate(x_test, y_test, verbose=0)
        accuracy = 100*score[1]

        print("Pre-training accuracy: %.4f%%" % accuracy)

        from keras.callbacks import ModelCheckpoint 
        from datetime import datetime 

        num_epochs = 100
        num_batch_size = 32

        checkpointer = ModelCheckpoint(filepath='../saved_models/weights.best.basic_mlp.hdf5', 
                                    verbose=1, save_best_only=True)
        start = datetime.now()

        model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

        # Evaluating the model on the training and testing set
        score = model.evaluate(x_train, y_train, verbose=0)
        print("Training Accuracy: ", score[1])

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Testing Accuracy: ", score[1])

        self.model= model
        self.le = le
    def getInstrumentTypes(self):
        return self.instrument_types

    def extract_feature(self, file_name):
        try:
            audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None, None

        return np.array([mfccsscaled])

    def extract_features(self, file_name):
    
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccsscaled = np.mean(mfccs.T,axis=0)
            
            print("mfccsscaled", mfccsscaled.shape)
            
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None 
        
        return mfccsscaled

    def get_prediction(self, file_name):
        prediction_feature = self.extract_feature(file_name) 

        predicted_vector = self.model.predict_classes(prediction_feature)
        predicted_class = self.le.inverse_transform(predicted_vector) 
        print("The predicted class is:", predicted_class[0], '\n') 

        predicted_proba_vector = self.model.predict_proba(prediction_feature) 
        predicted_proba = predicted_proba_vector[0]
        for i in range(len(predicted_proba)): 
            category = self.le.inverse_transform(np.array([i]))
            print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
        
        return predicted_class[0]

    def classify(self, filename):
        return self.get_prediction(filename)