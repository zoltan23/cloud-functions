from instrument_classification_module import InstrumentClassification

ic = InstrumentClassification()

def predict(filename, instrument):
    predicted_instrument = ic.classify(filename)
    if (instrument == predicted_instrument):
        print('TEST PASSED!!!!! Expected' + instrument + ' and got ' + predicted_instrument )
    else:
        print('TEST FAILED!!!!! Expected' + instrument + ' and got ' + predicted_instrument )


def predictedTrumpetTest1():
    predict('../audio-files/long-tone-1.wav', "Trumpet")
def predictedTrumpetTest2():
    predict('../audio-files/long-tone-2.wav', "Trumpet")
def predictedTrumpetTest3():
    predict('../audio-files/long-tone-3.wav', "Trumpet")

def predictedTrumpetTest4():
    predict('../audio-files/long-tone-3-chunk0.wav', "Trumpet")


# predictedTrumpetTest1()
# predictedTrumpetTest2()
# predictedTrumpetTest3()
predictedTrumpetTest4()

    

# predictedInstrumentTest('/tf/desktop/Audio-Classification/clean/229be2be.wav', "Saxophone")
# predictedInstrumentTest('/tf/desktop/Audio-Classification/clean/229be2be.wav', "Clarinet")
