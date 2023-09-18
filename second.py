import pyaudio
import numpy
print("hello")
RATE=16000
RECORD_SECONDS = 30
CHUNKSIZE = 1024
# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
frames = [] # A python-list of chunks(numpy.ndarray)
for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
 data = stream.read(CHUNKSIZE)
 frames.append(numpy.fromstring(data, dtype=numpy.int16))
#Convert the list of numpy-arrays into a 1D array (column-wise)
numpydata = numpy.hstack(frames)
# close stream
stream.stop_stream()
stream.close()
p.terminate()
import scipy.io.wavfile as wav
wav.write('out.wav',RATE,numpydata)