import librosa
import soundfile
import os, glob, pickle
import pyttsx3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

"""
Here,I have defined a function extract_feature to extract the 
mfcc, chroma, and mel features from a sound file. 
This function takes 4 parameters- the file name 
and three Boolean parameters for the three features:

"""
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

"""
Here, I defined a dictionary to hold numbers 
and the emotions available in the RAVDESS dataset

"""
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

# Here, I created a list of emotions that has to be detected from the voice.

observed_emotions=[ 'happy', 'fearful', 'angry','surprised']

"""
Here, I created a fuction load_data() – this takes in the relative 
size of the test set as parameter. x and y are empty lists; 
I used the glob() function from the glob module to
get all the pathnames for the sound files in our dataset. 
Here I used the Ravdess data set which contain 24 folders of different 
voice with 60 audio files in each folder

"""


def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("Actor_\\.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)




file="Actor_02/03-01-03-01-01-02-02.wav"

"""
Here I collected a sound file in the variable file
for which I use this model to detect in the emotion 
in the file and then passed that file to extract_fea
ture function

"""
#file="out.wav"
feature=extract_feature(file,mfcc=True,chroma=True,mel=True)





x_train,x_test,y_train,y_test=load_data(test_size=0.2)





# print((x_train.shape[0], x_test.shape[0]))



# print(f'Features extracted: {x_train.shape[1]}')



model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pre=model.predict([feature])
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)


#print("Accuracy: {:.2f}%".format(accuracy*100))



"""
Here I have taken a varible speaker to use it as speaker 
which speak our obsereved emotion and accuracy of the model

"""
speaker = pyttsx3.init()


"""
Here I have created funtion a talk which is used to talking purpose
in the code we just have to pass the line or whatever we want to speak 

"""
def talk(text):
    speaker.say(text)
    speaker.runAndWait()



#print(y_pre)
talk(' the obeserved emotion from the given audio file is ')
talk(y_pre)
talk(' and the accuracy of detection of emotion is  ')
accu=int(accuracy*100)
talk(accu)
talk('%')