'''
import numpy as np
import simpleaudio as sa
import matplotlib.pyplot as plt
# calculate note frequencies Hertz
A_freq = 120
Csh_freq = A_freq * 2 ** (4 / 12)

E_freq = A_freq * 2 ** (7 / 12)
print(Csh_freq)
print(E_freq)
# get timesteps for each sample, T is note duration in seconds
sample_rate = 44100
T = 0.25
t = np.linspace(0, T, T * sample_rate, False)
#print(t)
# generate sine wave notes
A_note = np.sin(A_freq * t * 2 * np.pi)
Csh_note = np.sin(Csh_freq * t * 2 * np.pi)
E_note = np.sin(E_freq * t * 2 * np.pi)

plt.subplot(211)
plt.plot(A_note)
# concatenate notes
audio = np.hstack((A_note, Csh_note,E_note))
#print(audio)
# normalize to 16-bit range
audio *= 32767 / 1 * np.max(np.abs(audio))
plt.subplot(212)
#print(audio)
# convert to 16-bit data
audio = audio.astype(np.int16)

# create stereo signal with zeros
stereo_signal = np.zeros([int(sample_rate*T*3),2],dtype=np.int16)   
stereo_signal[:,1] = audio[:]
print(stereo_signal)

#print(audio2)
plt.plot(audio)
# start playback
play_obj = sa.play_buffer(stereo_signal, 2, 2, sample_rate)

#plt.show()
# wait for playback to finish before exiting
play_obj.wait_done()
'''



import numpy as np
import simpleaudio as sa

# calculate note frequencies
A_freq = 600
Csh_freq = A_freq * 2 ** (4 / 12)
E_freq = A_freq * 2 ** (7 / 12)

# get timesteps for each sample, T is note duration in seconds
sample_rate = 44100
T = 0.15
t = np.linspace(0, T, T * sample_rate, False)

# generate sine wave notes
A_note = np.sin(A_freq * t * 2 * np.pi)
Csh_note = np.sin(Csh_freq * t * 2 * np.pi)
E_note = np.sin(E_freq * t * 2 * np.pi)

# mix audio together
audio = np.zeros((44100, 2))
n = len(t)
offset = 0

# 1 right side, 0 left side

audio[0 + offset: n + offset, 0] += 0 * A_note
audio[0 + offset: n + offset, 1] += 1 * A_note
offset = 5500
audio[0 + offset: n + offset, 0] += 0 * A_note
audio[0 + offset: n + offset, 1] += 1 * A_note
offset = 11000
audio[0 + offset: n + offset, 0] += 0 * A_note
audio[0 + offset: n + offset, 1] += A_note

# normalize to 16-bit range
audio *= 32767 / np.max(np.abs(audio))
# convert to 16-bit data
audio = audio.astype(np.int16)

# start playback
play_obj = sa.play_buffer(audio, 2, 2, sample_rate)

# wait for playback to finish before exiting
play_obj.wait_done()