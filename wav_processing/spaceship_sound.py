import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
from scipy.signal import medfilt, butter, lfilter

# ==========================================================
# Configuration
# ==========================================================
INPUT_WAV = "audio.wav"
OUTPUT_WAV = "space.wav"

SR = 22050
HOP_LENGTH = 512
FRAME_DURATION = HOP_LENGTH / SR

MIN_FREQ = 80
MAX_FREQ = 1000

PITCH_TOLERANCE = 0.5      # semitones
MEDIAN_FILTER_SIZE = 7

# ==========================================================
# Utility functions
# ==========================================================
def hz_to_midi(f):
    return 69 + 12 * np.log2(f / 440.0)

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12))

def lowpass(signal, cutoff=2500, sr=22050):
    b, a = butter(2, cutoff / (sr / 2), btype="low")
    return lfilter(b, a, signal)

# ==========================================================
# Simple, safe keyboard-like tone (frame-based)
# ==========================================================
def simple_tone(freq, duration, vel, sr):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Basic harmonic tone
    signal = (
        np.sin(2 * np.pi * freq * t) +
        0.3 * np.sin(2 * np.pi * 2 * freq * t)
    )

    # Envelope sizes RELATIVE to frame length
    attack = min(int(0.2 * n), n)
    release = min(int(0.3 * n), n)

    env = np.ones(n)

    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        env[-release:] = np.linspace(1, 0, release)

    return vel * env * signal

# ==========================================================
# Load audio
# ==========================================================
audio, sr = librosa.load(INPUT_WAV, sr=SR, mono=True)

# RMS → velocity
rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
rms = rms / (np.max(rms) + 1e-6)

# ==========================================================
# Pitch tracking
# ==========================================================
f0, _, _ = librosa.pyin(
    audio,
    fmin=MIN_FREQ,
    fmax=MAX_FREQ,
    sr=sr,
    hop_length=HOP_LENGTH
)

midi = np.array([
    hz_to_midi(f) if f is not None else 0.0
    for f in f0
])

# Smooth pitch to reduce jumpiness
midi_smooth = medfilt(midi, MEDIAN_FILTER_SIZE)

# ==========================================================
# Frame-based resynthesis (time preserved)
# ==========================================================
output = np.zeros_like(audio, dtype=np.float32)
current_pitch = None

for i, m in enumerate(midi_smooth):
    start = i * HOP_LENGTH
    end = start + HOP_LENGTH
    if end > len(output):
        break

    if m > 0:
        if current_pitch is None or abs(m - current_pitch) > PITCH_TOLERANCE:
            current_pitch = m

        freq = midi_to_hz(round(current_pitch))
        vel = rms[i]

        grain = simple_tone(freq, FRAME_DURATION, vel, sr)
        output[start:end] += grain[:HOP_LENGTH]

    else:
        current_pitch = None

# ==========================================================
# Post-processing
# ==========================================================
output = lowpass(output, sr=sr)
output /= np.max(np.abs(output) + 1e-6)

# ==========================================================
# Save and play
# ==========================================================
sf.write(OUTPUT_WAV, output, sr)

print("Playing output...")
sd.play(output, sr)
sd.wait()

print(f"Done. Saved as '{OUTPUT_WAV}'")
