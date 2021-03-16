import numpy as np
import librosa
import preprocessing
import classify


def feature_extract(audio_filename):
    sr = 44100
    window_size = 2048
    hop_size = window_size / 2
    #loading the music file
    music, sr = librosa.load(audio_filename, sr=sr)
    start_trim = preprocessing.detect_leading_silence(music)
    end_trim = preprocessing.detect_leading_silence(np.flipud(music))

    duration = len(music)
    #trimming the sound
    trimmed_sound = music[start_trim:duration - end_trim]
    mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
    avg = np.mean(mfccs, axis=1)
    audio_feature = avg.reshape(20)
    return audio_feature

#deciding the results based on prediction
def result(pre):
    if pre == 1:
        print("The prediction of this instrument is mandolin.")
    elif pre == 2:
        print("The prediction of this instrument is guitar.")
    elif pre == 3:
        print("The prediction of this instrument is trumpet.")


def main():
    #input a file name
    audio_filename = "final_data/data/guitar/guitar_As2_very-long_forte_normal.mp3"
    demo_data = feature_extract(audio_filename)
    demo_data = np.array([demo_data])
    #passing the parameters for pca
    model_params = {
        'pca_n': 10,
    }
    #loading the saved model
    model = classify.load_model(model_params)
    #getting predictions from the loaded model
    pre = classify.predict(model, demo_data, [model_params, 'pca'])
    result(pre)


if __name__ == '__main__':
    main()
