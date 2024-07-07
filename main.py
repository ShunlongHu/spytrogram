import numpy as np
import matplotlib.pyplot as plt
import threading
import pyaudio
import time
import scipy.signal

maxFreq = 10000
minFreq = 20
height = 1024
width = 100
displayTime = 10
interval = 1000 // (width/displayTime)
maxDb=0
minDb=-80

data = np.ones(height)*minDb
display = np.ones((height, width))*minDb
freqCoord = ((maxFreq/minFreq)**np.linspace(0, 1, height + 1)[:-1]) * minFreq

freqPoint = []
freqIdx = []
for f in range(minFreq, maxFreq):
    fStr = str(f)
    if sum([i != '0' for i in fStr]) == 1:
        freqPoint += [f if fStr[0] == '1' else '']
        idx = np.log10(f / minFreq) / np.log10((maxFreq/minFreq)) * height
        freqIdx += [idx]
freqPoint += [str(maxFreq)]
freqIdx += [height]
print(freqIdx)
print(freqPoint)

tonePoint = ['']
toneIdx = [0]
for t in range(-3, 4):
    for p in [0, 2, 4, 5, 7, 9, 11]:
        tone = t * 12 + p
        baseTone = 440 * 2**(3/12) / 2
        f = baseTone * 2 **(tone/12)
        if tone % 12 == 0:
            tonePoint += ['c' + str(tone // 12 + 4)]
        else:
            tonePoint += ['']
        print(f)
        toneIdx += [np.log10(f / minFreq) / np.log10((maxFreq/minFreq)) * height]

tonePoint += ['']
toneIdx += [height]
print(tonePoint)
print(toneIdx)
print(((maxFreq/minFreq)**(np.array(toneIdx)/height) * minFreq))
print(freqIdx)
print(((maxFreq/minFreq)**(np.array(freqIdx)/height) * minFreq))
def updateRecordData():
    global data
    samplingRate = 48000
    chunk = 1024
    cacheSize = 8196

    freqDiff = freqCoord * 1.0
    freqDiff[1:] = (freqCoord[1:] - freqCoord[:-1]) / 2
    freqDiff[0] = freqCoord[1] - freqCoord[0]
    kernal = np.zeros((height, cacheSize), complex)
    hanning=np.hanning(cacheSize)
    for idx in range(height):
        # kernal[idx] = scipy.signal.firwin(cacheSize, [freqCoord[idx] - freqDiff[idx], freqCoord[idx] + freqDiff[idx]], pass_zero=False, fs=samplingRate)
        kernal[idx] = np.exp(-1j*2*np.pi*freqCoord[idx]/samplingRate*np.arange(cacheSize)) / cacheSize * hanning
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=samplingRate, frames_per_buffer=chunk, input=True)
    cache = np.zeros(cacheSize)
    while True:
        audio = stream.read(chunk)
        cache[:cacheSize-chunk] = cache[chunk:]
        cache[cacheSize-chunk:] = np.frombuffer(audio, dtype=np.float32)
        data = np.log10(np.abs(kernal.dot(cache)))*10

recorder = threading.Thread(target=updateRecordData)
recorder.start()

fig = plt.figure()
ax=plt.imshow(display, interpolation="bilinear", vmax=maxDb, vmin=minDb, cmap="magma", aspect="auto")
plt.gca().invert_yaxis()
plt.ion()
plt.show()
plt.grid()
plt.yticks(freqIdx, freqPoint)
plt.gca().twinx()
plt.yticks(toneIdx, tonePoint)
# plt.grid()
frame = 0
while True:
    startTime = time.time()
    display[:,:-1] = display[:,1:]
    display[:,-1] = data
    ax.set_data(display)

    dotsPerSec = width // displayTime
    xStart = (-frame) % dotsPerSec
    xPos = np.arange(displayTime) * dotsPerSec + xStart

    plt.xticks(xPos, np.arange(displayTime) + frame // dotsPerSec + (-1 if frame % dotsPerSec == 0 else 0))
    frame += 1
    plt.draw()
    stopTime = time.time()
    if frame % (width // displayTime) == 0:
        print(startTime)
    plt.pause(max(0, interval/1000-(stopTime - startTime)))
