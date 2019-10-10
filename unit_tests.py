import instrument_classification_module

#Create a shortened module reference
mod = instrument_classification_module

filename = '/tf/desktop/Audio-Classification/clean/229be2be.wav' 

#Unit tests
mod.read_wavfile(filename)

#Initialize variables for testing
librosa_audio, scipy_audio, librosa_sample_rate, scipy_sample_rate = mod.read_wavfile(filename)

mod.wavfile_min_max(scipy_audio, librosa_audio)
mod.plot_wavfile(scipy_audio)
mod.plot_merged(librosa_audio)
mod.mffcs(librosa_audio, librosa_sample_rate)



