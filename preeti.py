import librosa
import soundfile
import os, glob, pickle
import pyttsx3
import numpy as np
import pyaudio
import numpy
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



import warnings
warnings.filterwarnings("ignore")


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
            return result


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
speaker = pyttsx3.init()
speaker.setProperty("rate", 100)


listener = sr.Recognizer()
def talk(text):
    speaker.say(text)
    speaker.runAndWait()



observed_emotions=[ 'happy', 'fearful', 'angry','surprised']






def get_info() :      
    with sr.Microphone() as source:
        print('listening...')
        voice = listener.record(source,duration=4)

        try:
            info = listener.recognize_google(voice)
            
            return info.lower()

        except:
            talk('did not hear properly')
            get_info()


def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)




talk("if you want to test the model on file from system speak system else speak live for recording your audio");

option=get_info()
print(option)

if option != "live":
    file="Actor_04/03-01-01-01-01-01-04.wav"
    
else:
    print("hello")
    RATE=16000
    RECORD_SECONDS = 15
    CHUNKSIZE = 1024
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
    frames = []
    for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
     data = stream.read(CHUNKSIZE)
     frames.append(numpy.fromstring(data, dtype=numpy.int16))
    
    numpydata = numpy.hstack(frames)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    import scipy.io.wavfile as wav
    wav.write('out.wav',RATE,numpydata)
    file="out.wav"



    
    



feature=extract_feature(file,mfcc=True,chroma=True,mel=True)





x_train,x_test,y_train,y_test=load_data(test_size=0.2)




model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pre=model.predict([feature])
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)


speaker = pyttsx3.init()


print(y_pre)
talk(' the obeserved emotion from the given audio file is ')
talk(y_pre)
talk(' and the accuracy of detection of emotion is  ')
accu=int(accuracy*100)
print(accu)
talk(accu)
talk('%')


