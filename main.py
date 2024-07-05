import numpy as np
import matplotlib.pyplot as plt
import threading
import pyaudio
import time

maxFreq = 10000
minFreq = 20
height = 1024
width = 100
displayTime = 5
interval = 1000 // (width/displayTime)
maxDb=10
minDb=-60

data = np.ones(height)*minDb
display = np.ones((height, width))*minDb
freqCoord = ((maxFreq/minFreq)**np.linspace(0, 1, height)) * minFreq

def updateRecordData():
    global data
    samplingRate = 48000
    chunk = 1024
    cacheSize = 8196

    xp = np.linspace(0, samplingRate/2, cacheSize//2)
    x = freqCoord

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=samplingRate, frames_per_buffer=chunk, input=True)
    cache = np.zeros(cacheSize)
    hanning=np.hanning(cacheSize)
    while True:
        audio = stream.read(chunk)
        cache[:cacheSize-chunk] = cache[chunk:]
        cache[cacheSize-chunk:] = np.frombuffer(audio, dtype=np.float32)
        freq = np.log10(np.abs(np.fft.fft(cache*hanning)[:cacheSize//2]))*10
        data = np.interp(x, xp, freq)

recorder = threading.Thread(target=updateRecordData)
recorder.start()

fig = plt.figure()
ax=plt.imshow(display, interpolation="bilinear", vmax=maxDb, vmin=minDb, cmap="magma", aspect="auto")
plt.gca().invert_yaxis()
plt.ion()
plt.show()
plt.grid()
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
    plt.yticks(np.arange(10) * 100)
    frame += 1
    plt.draw()

    plt.pause(0.001)
    stopTime = time.time()
    if frame % (width // displayTime) == 0:
        print(startTime)
    time.sleep(max(0, interval/1000-(stopTime - startTime)))
