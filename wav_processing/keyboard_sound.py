import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
from scipy.signal import medfilt, butter, lfilter

# ----------------------------
# Parameters
# ----------------------------
INPUT_WAV = "audio.wav"
OUTPUT_WAV = "playback.wav"

SR = 22050
HOP_LENGTH = 512

MIN_FREQ = 80
MAX_FREQ = 1000

MIN_NOTE_LEN = 0.15
PITCH_TOLERANCE = 0.5  # semitones
MEDIAN_FILTER_SIZE = 7

# ----------------------------
# Utility functions
# ----------------------------
def hz_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12))

def lowpass(signal, cutoff=3000, sr=22050):
    b, a = butter(2, cutoff / (sr / 2), btype='low')
    return lfilter(b, a, signal)

# ----------------------------
# Keyboard-style synth (softer)
# ----------------------------
def keyboard_synth(freq, duration, velocity, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Gentle additive synth
    signal = (
        np.sin(2 * np.pi * freq * t) +
        0.3 * np.sin(2 * np.pi * 2 * freq * t) +
        0.15 * np.sin(2 * np.pi * 3 * freq * t)
    )

    # Softer ADSR
    attack = int(0.08 * sr)
    release = int(0.15 * sr)

    env = np.ones_like(signal)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)

    signal *= env * velocity
    return lowpass(signal, sr=sr)

# ----------------------------
# Load audio
# ----------------------------
audio, sr = librosa.load(INPUT_WAV, sr=SR, mono=True)

# RMS energy (for velocity)
rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
rms = np.clip(rms / np.max(rms), 0, 1)

# ----------------------------
# Pitch tracking
# ----------------------------
f0, voiced, _ = librosa.pyin(
    audio,
    fmin=MIN_FREQ,
    fmax=MAX_FREQ,
    sr=sr,
    hop_length=HOP_LENGTH
)

# Convert to MIDI and smooth
midi = np.array([
    hz_to_midi(f) if f is not None else np.nan
    for f in f0
])

midi_smooth = medfilt(
    np.nan_to_num(midi, nan=0),
    kernel_size=MEDIAN_FILTER_SIZE
)

times = librosa.frames_to_time(
    np.arange(len(midi_smooth)),
    sr=sr,
    hop_length=HOP_LENGTH
)

# ----------------------------
# Improved note segmentation
# ----------------------------
notes = []
current_pitch = None
start_time = None
velocities = []

for i, (t, m) in enumerate(zip(times, midi_smooth)):
    if m > 0:
        if current_pitch is None:
            current_pitch = m
            start_time = t
            velocities = [rms[i]]
        elif abs(m - current_pitch) <= PITCH_TOLERANCE:
            velocities.append(rms[i])
        else:
            duration = t - start_time
            if duration >= MIN_NOTE_LEN:
                notes.append((
                    midi_to_hz(round(current_pitch)),
                    duration,
                    np.mean(velocities)
                ))
            current_pitch = m
            start_time = t
            velocities = [rms[i]]
    else:
        if current_pitch is not None:
            duration = t - start_time
            if duration >= MIN_NOTE_LEN:
                notes.append((
                    midi_to_hz(round(current_pitch)),
                    duration,
                    np.mean(velocities)
                ))
            current_pitch = None

# ----------------------------
# Resynthesis
# ----------------------------
output = np.array([], dtype=np.float32)

for freq, dur, vel in notes:
    output = np.concatenate(
        (output, keyboard_synth(freq, dur, vel, sr))
    )

# Normalise
output /= np.max(np.abs(output) + 1e-6)

# Save & play
sf.write(OUTPUT_WAV, output, sr)

print("Playing improved keyboard sound...")
sd.play(output, sr)
sd.wait()

print("Done.")
