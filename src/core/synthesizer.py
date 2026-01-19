"""
Synthesizer module for UPIC.

This is the real-time audio engine that converts arcs into sound.
It performs wavetable synthesis with optional frequency modulation,
applying envelopes and mixing multiple voices.

The original UPIC used 64 simultaneous oscillators at 44.1kHz.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict
from threading import Thread, Event
import queue

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from .waveform import Waveform, WaveformLibrary
from .envelope import Envelope, EnvelopeLibrary
from .frequency_table import FrequencyTable, FrequencyTableLibrary
from .arc import Arc
from .page import Page


@dataclass
class AudioEffects:
    """
    Lightweight post-processing effects applied to the mixed signal.

    IMPORTANT:
    - Must be fast and safe for real-time audio callback.
    - Defaults are intentionally mild.
    """
    enabled: bool = True

    # Simple EQ (1-pole HP/LP)
    highpass_hz: float = 30.0
    lowpass_hz: float = 16000.0

    # Phaser (disabled by default; stateful + more CPU)
    phaser_enabled: bool = False
    phaser_rate_hz: float = 0.25
    phaser_min_hz: float = 300.0
    phaser_max_hz: float = 1500.0
    phaser_feedback: float = 0.0
    phaser_mix: float = 0.0  # 0 = dry, 1 = wet
    phaser_stages: int = 4

    # Distortion (optional; additional to final soft-clip limiter)
    distortion_enabled: bool = False
    distortion_drive: float = 1.0
    distortion_mix: float = 0.0  # 0 = dry, 1 = wet
    distortion_mode: str = "tanh"  # "tanh" | "hardclip" | "guitar"
    distortion_clip: float = 0.25  # Used for hardclip (smaller = more extreme)
    distortion_oversample: int = 1  # 1, 2, or 4 (higher = less aliasing, more CPU)

    # Post-FX safety level (linear scaling, not extra distortion)
    output_ceiling: float = 0.9
    auto_level: bool = True

    # "Fuzz" voicing helpers (optional)
    # Bass boost is applied as a low-shelf (biquad) around bass_boost_hz.
    bass_boost_enabled: bool = False
    bass_boost_db: float = 0.0
    bass_boost_hz: float = 140.0
    bass_boost_slope: float = 0.8  # 0.1..2 (1.0 is typical)

    # Post-distortion tone control (LP after distortion to get fuzzier / less harsh).
    fuzz_tone_lowpass_hz: float = 0.0  # 0 disables


@dataclass
class Voice:
    """
    A single synthesis voice (oscillator).
    
    Each voice can play one arc at a time, with its own
    phase accumulator for wavetable lookup.
    """
    arc: Optional[Arc] = None
    waveform: Optional[Waveform] = None
    envelope: Optional[Envelope] = None
    frequency_table: Optional[FrequencyTable] = None
    
    # Synthesis state
    phase: float = 0.0  # Current phase (0-1)
    active: bool = False
    
    # Modulation
    modulator_voice: Optional[Voice] = None
    
    def reset(self) -> None:
        """Reset voice state."""
        self.phase = 0.0
        self.active = False
        self.arc = None
        self.modulator_voice = None
        self._rendering = False
    
    def start(
        self,
        arc: Arc,
        waveform: Waveform,
        envelope: Envelope,
        frequency_table: FrequencyTable
    ) -> None:
        """Start playing an arc."""
        self.arc = arc
        self.waveform = waveform
        self.envelope = envelope
        self.frequency_table = frequency_table
        self.phase = 0.0
        self.active = True
    
    # Track rendering to prevent recursion
    _rendering: bool = False
    
    def render(
        self,
        current_time: float,
        num_samples: int,
        sample_rate: float
    ) -> NDArray[np.float64]:
        """
        Render audio samples for this voice.
        
        Args:
            current_time: Current playback time in seconds
            num_samples: Number of samples to generate
            sample_rate: Audio sample rate
            
        Returns:
            Array of audio samples
        """
        # Prevent infinite recursion from circular modulation
        if self._rendering:
            return np.zeros(num_samples)
        
        if not self.active or self.arc is None:
            return np.zeros(num_samples)
        
        if self.waveform is None or self.envelope is None or self.frequency_table is None:
            return np.zeros(num_samples)
        
        # Check if arc is active at this time
        if current_time > self.arc.end_time:
            self.active = False
            return np.zeros(num_samples)
        
        if current_time + num_samples / sample_rate < self.arc.start_time:
            return np.zeros(num_samples)
        
        # Mark as rendering to prevent recursion
        self._rendering = True
        
        # Generate time array for this buffer
        times = current_time + np.arange(num_samples) / sample_rate
        
        # Get pitch positions at each time point
        pitch_positions = self.arc.get_pitches_at_times(times)
        
        # Convert pitch positions to frequencies
        frequencies = self.frequency_table.get_frequencies_at_positions(pitch_positions)
        
        # Apply frequency modulation if modulator is set
        if self.modulator_voice is not None and self.modulator_voice.active:
            mod_signal = self.modulator_voice.render(current_time, num_samples, sample_rate)
            mod_index = self.arc.modulation_index
            frequencies = frequencies * (1 + mod_index * mod_signal)

        # Safety clamp: keep frequencies within a sensible, playable range.
        # This prevents FM from driving frequencies sub-audio, negative, or ultrasonically high.
        min_allowed = max(1.0, float(self.frequency_table.min_freq))
        nyquist_guard = float(sample_rate) * 0.45  # stay below Nyquist to reduce aliasing
        max_allowed = min(float(self.frequency_table.max_freq), nyquist_guard)
        if max_allowed <= min_allowed:
            max_allowed = min_allowed + 1.0
        frequencies = np.clip(frequencies, min_allowed, max_allowed)
        
        # Calculate phase increments
        phase_increments = frequencies / sample_rate
        
        # Accumulate phase
        phases = np.zeros(num_samples)
        phase = self.phase
        for i in range(num_samples):
            phases[i] = phase
            phase += phase_increments[i]
            phase = phase % 1.0
        self.phase = phase
        
        # Look up waveform values
        samples = self.waveform.get_samples_at_phases(phases)
        
        # Calculate envelope position (normalized within arc duration)
        arc_duration = self.arc.duration
        if arc_duration > 0:
            envelope_positions = (times - self.arc.start_time) / arc_duration
            envelope_positions = np.clip(envelope_positions, 0, 1)
        else:
            envelope_positions = np.zeros(num_samples)
        
        # Apply envelope
        envelope_values = self.envelope.get_amplitudes_at_positions(envelope_positions)
        samples = samples * envelope_values
        
        # Apply per-arc envelope if defined
        if self.arc.envelope_points:
            for i, t in enumerate(times):
                samples[i] *= self.arc.get_envelope_amplitude_at_time(t)
        
        # Apply arc amplitude and create mask for active region
        samples = samples * self.arc.amplitude
        
        # Zero out samples outside arc time range
        mask = (times >= self.arc.start_time) & (times <= self.arc.end_time)
        samples = samples * mask
        
        # Done rendering
        self._rendering = False
        
        return samples


@dataclass
class Synthesizer:
    """
    The main UPIC synthesizer engine.
    
    Manages multiple voices and handles real-time audio output.
    
    Attributes:
        sample_rate: Audio sample rate (default 44100 Hz)
        buffer_size: Audio buffer size in samples
        num_voices: Maximum number of simultaneous voices
    """
    sample_rate: int = 44100
    buffer_size: int = 2048  # Larger buffer to prevent audio glitches
    num_voices: int = 64
    
    # Libraries
    waveforms: WaveformLibrary = field(default_factory=WaveformLibrary)
    envelopes: EnvelopeLibrary = field(default_factory=EnvelopeLibrary)
    frequency_tables: FrequencyTableLibrary = field(default_factory=FrequencyTableLibrary)
    
    # Voices
    voices: List[Voice] = field(default_factory=list)
    
    # Playback state
    playing: bool = False
    current_time: float = 0.0
    page: Optional[Page] = None
    
    # Audio stream
    _stream: Optional[sd.OutputStream] = None
    _stop_event: Event = field(default_factory=Event)
    
    # Master volume (reduced to prevent clipping)
    master_volume: float = 0.5

    # Post-FX (applied before final soft clip / output)
    effects: AudioEffects = field(default_factory=AudioEffects)

    # Effect state (kept in Synthesizer, not per Voice)
    _hp_x1: float = field(default=0.0, repr=False)
    _hp_y1: float = field(default=0.0, repr=False)
    _lp_y1: float = field(default=0.0, repr=False)
    _phaser_phase: float = field(default=0.0, repr=False)
    _phaser_x1: List[float] = field(default_factory=list, repr=False)
    _phaser_y1: List[float] = field(default_factory=list, repr=False)
    _phaser_fb: float = field(default=0.0, repr=False)

    # Bass shelf state (biquad)
    _bass_x1: float = field(default=0.0, repr=False)
    _bass_x2: float = field(default=0.0, repr=False)
    _bass_y1: float = field(default=0.0, repr=False)
    _bass_y2: float = field(default=0.0, repr=False)
    
    # Callbacks
    on_time_update: Optional[Callable[[float], None]] = None
    on_playback_stop: Optional[Callable[[], None]] = None
    
    def __post_init__(self) -> None:
        """Initialize voices."""
        self.voices = [Voice() for _ in range(self.num_voices)]
        # Initialize phaser states
        self._phaser_x1 = [0.0 for _ in range(max(1, int(self.effects.phaser_stages)))]
        self._phaser_y1 = [0.0 for _ in range(max(1, int(self.effects.phaser_stages)))]

    def reset_effects_state(self) -> None:
        """Reset effect state (useful when seeking/looping)."""
        self._hp_x1 = 0.0
        self._hp_y1 = 0.0
        self._lp_y1 = 0.0
        self._phaser_phase = 0.0
        self._phaser_fb = 0.0
        stages = max(1, int(self.effects.phaser_stages))
        self._phaser_x1 = [0.0 for _ in range(stages)]
        self._phaser_y1 = [0.0 for _ in range(stages)]
        self._bass_x1 = 0.0
        self._bass_x2 = 0.0
        self._bass_y1 = 0.0
        self._bass_y2 = 0.0

    def _apply_low_shelf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Low-shelf filter (biquad), stateful.

        Uses RBJ Audio EQ Cookbook coefficients.
        """
        if not self.effects.bass_boost_enabled:
            return x
        if abs(self.effects.bass_boost_db) < 1e-6:
            return x

        fs = float(self.sample_rate)
        f0 = float(np.clip(self.effects.bass_boost_hz, 10.0, fs * 0.45))
        S = float(np.clip(self.effects.bass_boost_slope, 0.1, 2.0))
        A = float(10.0 ** (float(self.effects.bass_boost_db) / 40.0))

        w0 = 2.0 * np.pi * f0 / fs
        cos_w0 = float(np.cos(w0))
        sin_w0 = float(np.sin(w0))
        alpha = float(sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0))
        sqrtA = float(np.sqrt(A))

        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrtA * alpha
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrtA * alpha

        # Normalize
        b0 /= a0
        b1 /= a0
        b2 /= a0
        a1 /= a0
        a2 /= a0

        y = np.empty_like(x)
        x1 = self._bass_x1
        x2 = self._bass_x2
        y1 = self._bass_y1
        y2 = self._bass_y2
        for i in range(len(x)):
            xn = float(x[i])
            yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1
            x1 = xn
            y2 = y1
            y1 = yn
            y[i] = yn
        self._bass_x1 = float(x1)
        self._bass_x2 = float(x2)
        self._bass_y1 = float(y1)
        self._bass_y2 = float(y2)
        return y

    def _apply_one_pole_highpass(self, x: NDArray[np.float64], cutoff_hz: float) -> NDArray[np.float64]:
        """
        1-pole high-pass filter (stateful).

        y[n] = a * (y[n-1] + x[n] - x[n-1])
        where a = exp(-2*pi*fc/fs)
        """
        if cutoff_hz <= 0:
            return x

        fs = float(self.sample_rate)
        a = float(np.exp(-2.0 * np.pi * float(cutoff_hz) / fs))

        y = np.empty_like(x)
        x1 = self._hp_x1
        y1 = self._hp_y1
        for i in range(len(x)):
            y1 = a * (y1 + x[i] - x1)
            x1 = x[i]
            y[i] = y1
        self._hp_x1 = float(x1)
        self._hp_y1 = float(y1)
        return y

    def _apply_one_pole_lowpass(self, x: NDArray[np.float64], cutoff_hz: float) -> NDArray[np.float64]:
        """
        1-pole low-pass filter (stateful).

        y[n] = (1-a) * x[n] + a * y[n-1]
        where a = exp(-2*pi*fc/fs)
        """
        if cutoff_hz <= 0:
            return x

        nyquist = float(self.sample_rate) * 0.5
        cutoff_hz = float(min(cutoff_hz, nyquist * 0.99))
        fs = float(self.sample_rate)
        a = float(np.exp(-2.0 * np.pi * float(cutoff_hz) / fs))

        y = np.empty_like(x)
        y1 = self._lp_y1
        for i in range(len(x)):
            y1 = (1.0 - a) * x[i] + a * y1
            y[i] = y1
        self._lp_y1 = float(y1)
        return y

    def _allpass_coeff_from_hz(self, freq_hz: float) -> float:
        """Convert center frequency to 1st-order all-pass coefficient."""
        fs = float(self.sample_rate)
        freq_hz = float(np.clip(freq_hz, 1.0, fs * 0.45))
        w = np.tan(np.pi * freq_hz / fs)
        # a in (-1, 1) for stability
        return float((1.0 - w) / (1.0 + w))

    def _apply_phaser(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Simple multi-stage all-pass phaser (stateful)."""
        if not self.effects.phaser_enabled or self.effects.phaser_mix <= 0:
            return x

        stages = max(1, int(self.effects.phaser_stages))
        if len(self._phaser_x1) != stages or len(self._phaser_y1) != stages:
            self._phaser_x1 = [0.0 for _ in range(stages)]
            self._phaser_y1 = [0.0 for _ in range(stages)]

        # LFO for center frequency sweep
        fs = float(self.sample_rate)
        rate = float(max(0.0, self.effects.phaser_rate_hz))
        f_min = float(max(1.0, self.effects.phaser_min_hz))
        f_max = float(max(f_min, self.effects.phaser_max_hz))
        fb = float(np.clip(self.effects.phaser_feedback, -0.95, 0.95))
        mix = float(np.clip(self.effects.phaser_mix, 0.0, 1.0))

        y = np.empty_like(x)
        phase = float(self._phaser_phase)
        fb_state = float(self._phaser_fb)

        for i in range(len(x)):
            # LFO in [0,1]
            lfo = 0.5 * (1.0 + np.sin(2.0 * np.pi * phase))
            phase = (phase + rate / fs) % 1.0
            center_hz = f_min + (f_max - f_min) * lfo
            a = self._allpass_coeff_from_hz(center_hz)

            inp = x[i] + fb * fb_state
            stage_out = inp

            # Cascade 1st-order all-pass stages
            for s in range(stages):
                x1 = self._phaser_x1[s]
                y1 = self._phaser_y1[s]
                # y = -a*x + x1 + a*y1
                y_stage = (-a * stage_out) + x1 + (a * y1)
                self._phaser_x1[s] = float(stage_out)
                self._phaser_y1[s] = float(y_stage)
                stage_out = y_stage

            fb_state = stage_out
            wet = stage_out
            dry = x[i]
            y[i] = (1.0 - mix) * dry + mix * wet

        self._phaser_phase = float(phase)
        self._phaser_fb = float(fb_state)
        return y

    def _apply_distortion(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Distortion / fuzz (stateless)."""
        if not self.effects.distortion_enabled or self.effects.distortion_mix <= 0:
            return x
        drive = float(max(0.0, self.effects.distortion_drive))
        mix = float(np.clip(self.effects.distortion_mix, 0.0, 1.0))

        mode = str(getattr(self.effects, "distortion_mode", "tanh")).lower()
        oversample = int(getattr(self.effects, "distortion_oversample", 1))
        oversample = 1 if oversample not in (1, 2, 4) else oversample

        def shape(sig: NDArray[np.float64]) -> NDArray[np.float64]:
            if mode == "hardclip":
                clip = float(np.clip(getattr(self.effects, "distortion_clip", 0.25), 1e-3, 1.0))
                y = sig * drive
                y = np.clip(y, -clip, clip) / clip
                return y
            if mode == "guitar":
                # Multi-stage soft clip (more amp-like than raw hard clipping)
                y = np.tanh(sig * drive)
                y = np.tanh(y * (drive * 0.25 + 1.0))
                # Add a final gentle clamp to keep it controlled
                y = np.tanh(y * 1.5)
                return y
            # Default: smooth saturation
            return np.tanh(sig * drive)

        if oversample == 1:
            wet = shape(x)
        else:
            # Oversample around the nonlinearity to reduce aliasing ("digital clipping" harshness).
            # Prefer scipy if available (resample_poly does proper anti-aliasing).
            try:
                from scipy import signal as _signal  # type: ignore
                x_up = _signal.resample_poly(x, up=oversample, down=1)
                y_up = shape(x_up)
                wet = _signal.resample_poly(y_up, up=1, down=oversample)
                # resample_poly length can differ by +/-1; trim/pad to match input
                if len(wet) > len(x):
                    wet = wet[: len(x)]
                elif len(wet) < len(x):
                    wet = np.pad(wet, (0, len(x) - len(wet)))
            except Exception:
                # Fallback: linear up/downsample (still helps a bit)
                n = len(x)
                idx = np.linspace(0.0, 1.0, n, endpoint=False)
                idx_up = np.linspace(0.0, 1.0, n * oversample, endpoint=False)
                x_up = np.interp(idx_up, idx, x)
                y_up = shape(x_up)
                wet = y_up[::oversample]
                if len(wet) > n:
                    wet = wet[:n]
                elif len(wet) < n:
                    wet = np.pad(wet, (0, n - len(wet)))

        return (1.0 - mix) * x + mix * wet

    def _apply_effects(self, mixed: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply post-fx chain to mono mix (in-place safe)."""
        if not self.effects.enabled:
            return mixed

        y = mixed
        # EQ first
        if self.effects.highpass_hz > 0:
            y = self._apply_one_pole_highpass(y, self.effects.highpass_hz)
        if self.effects.lowpass_hz > 0:
            y = self._apply_one_pole_lowpass(y, self.effects.lowpass_hz)
        # Bass boost (low shelf) for thicker tone
        y = self._apply_low_shelf(y)
        # Optional phaser (kept, but default off)
        y = self._apply_phaser(y)
        # Optional distortion / fuzz
        y = self._apply_distortion(y)
        # Post-distortion tone control to make fuzz less harsh
        if self.effects.fuzz_tone_lowpass_hz and self.effects.fuzz_tone_lowpass_hz > 0:
            y = self._apply_one_pole_lowpass(y, self.effects.fuzz_tone_lowpass_hz)

        # Final: keep level under control without changing distortion character
        if getattr(self.effects, "auto_level", True):
            ceiling = float(np.clip(getattr(self.effects, "output_ceiling", 0.9), 0.05, 1.0))
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            if peak > ceiling and peak > 1e-12:
                y = y * (ceiling / peak)
        return y

    def _apply_output_limiter(self, mixed: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Final output stage.

        If we are already using a heavy nonlinear distortion, do NOT apply an extra tanh limiter
        (it reduces the visible/audible clipping character). We just clamp for safety.
        """
        if (
            getattr(self.effects, "enabled", False)
            and getattr(self.effects, "distortion_enabled", False)
            and float(getattr(self.effects, "distortion_mix", 0.0)) > 0.0
        ):
            return np.clip(mixed, -1.0, 1.0)

        # Default safety limiter for cleaner modes
        return np.tanh(mixed * 0.8) * 0.9
    
    def set_page(self, page: Page) -> None:
        """Set the page to synthesize."""
        self.page = page
    
    def play(self, start_time: float = 0.0) -> None:
        """Start playback from a given time."""
        if not AUDIO_AVAILABLE:
            print("Audio not available - sounddevice not installed")
            return
        
        if self.playing:
            self.stop()
        
        self.current_time = start_time
        self.playing = True
        self._stop_event.clear()
        
        # Debug info
        if self.page:
            print(f"Starting playback at {start_time:.2f}s")
            print(f"Page has {len(self.page.arcs)} arcs")
            for arc in self.page.arcs.values():
                print(f"  Arc '{arc.name}': {arc.start_time:.2f}-{arc.end_time:.2f}s, "
                      f"waveform={arc.waveform_name}, envelope={arc.envelope_name}")
        
        # Assign arcs to voices
        self._assign_voices()
        
        # Start audio stream with explicit dtype
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            blocksize=self.buffer_size,
            dtype='float32',
            callback=self._audio_callback
        )
        self._stream.start()
        print("Audio stream started")
    
    def stop(self) -> None:
        """Stop playback."""
        self.playing = False
        self._stop_event.set()
        
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Reset all voices
        for voice in self.voices:
            voice.reset()
        self.reset_effects_state()
        
        if self.on_playback_stop:
            self.on_playback_stop()
    
    def pause(self) -> None:
        """Pause playback (can be resumed)."""
        if self._stream is not None:
            self._stream.stop()
        self.playing = False
    
    def resume(self) -> None:
        """Resume paused playback."""
        if self._stream is not None:
            self._stream.start()
            self.playing = True
    
    def seek(self, time: float) -> None:
        """Seek to a specific time."""
        self.current_time = max(0, time)
        self._assign_voices()
        self.reset_effects_state()
    
    def _assign_voices(self) -> None:
        """Assign arcs to voices based on current time."""
        if self.page is None:
            return
        
        # Get active arcs
        active_arcs = self.page.get_active_arcs()
        active_arc_ids = {arc.id for arc in active_arcs}
        
        # Get default frequency table
        freq_table = self.frequency_tables.get(
            self.page.settings.frequency_table_name
        ) or self.frequency_tables.get("Continuous")
        
        # Track which arcs already have voices (don't reset them!)
        existing_arc_ids = set()
        for voice in self.voices:
            if voice.active and voice.arc is not None:
                if voice.arc.id in active_arc_ids:
                    existing_arc_ids.add(voice.arc.id)
                else:
                    # Arc is no longer active, reset this voice
                    voice.reset()
        
        # Find arcs that need new voices
        new_arcs = [arc for arc in active_arcs if arc.id not in existing_arc_ids]
        
        # Find available voices for new arcs
        arc_to_voice: Dict[str, Voice] = {}
        
        # First, map existing voices
        for voice in self.voices:
            if voice.active and voice.arc is not None:
                arc_to_voice[voice.arc.id] = voice
        
        # Assign new arcs to available voices
        for arc in new_arcs:
            # Find a free voice
            for voice in self.voices:
                if not voice.active:
                    waveform = self.waveforms.get(arc.waveform_name) or Waveform.sine()
                    envelope = self.envelopes.get(arc.envelope_name) or Envelope.adsr()
                    voice.start(arc, waveform, envelope, freq_table)
                    arc_to_voice[arc.id] = voice
                    break
        
        # Also assign voices to muted arcs that are used as modulators
        # (they need to be rendered even if not heard directly)
        for arc in self.page.arcs.values():
            if arc.muted and arc.id not in arc_to_voice:
                # Check if this muted arc is used as a modulator
                is_modulator = any(
                    other.modulator_id == arc.id 
                    for other in active_arcs
                )
                if is_modulator:
                    for voice in self.voices:
                        if not voice.active:
                            waveform = self.waveforms.get(arc.waveform_name) or Waveform.sine()
                            envelope = self.envelopes.get(arc.envelope_name) or Envelope.adsr()
                            voice.start(arc, waveform, envelope, freq_table)
                            arc_to_voice[arc.id] = voice
                            break
        
        # Set up modulation connections
        for arc in active_arcs:
            if arc.modulator_id and arc.id in arc_to_voice:
                modulator_voice = arc_to_voice.get(arc.modulator_id)
                if modulator_voice:
                    arc_to_voice[arc.id].modulator_voice = modulator_voice
    
    def _audio_callback(
        self,
        outdata: NDArray,
        frames: int,
        time_info,
        status
    ) -> None:
        """Audio stream callback - generates audio in real-time."""
        if status:
            print(f"Audio status: {status}")
        
        if not self.playing or self.page is None:
            outdata.fill(0)
            return
        
        # Update voice assignments (preserves phase for existing voices)
        self._assign_voices()
        
        # Mix all active voices
        mixed = np.zeros(frames, dtype=np.float64)
        active_count = 0
        
        for voice in self.voices:
            if voice.active:
                samples = voice.render(
                    self.current_time,
                    frames,
                    self.sample_rate
                )
                mixed += samples
                active_count += 1
        
        # Normalize by number of active voices to prevent clipping
        if active_count > 0:
            mixed = mixed / max(active_count, 1)
        
        # Apply master volume
        mixed = mixed * self.master_volume

        # Post-FX (EQ/phaser/distortion) BEFORE final limiter/output
        mixed = self._apply_effects(mixed)

        # Final output stage
        mixed = self._apply_output_limiter(mixed)
        
        # Output as stereo (ensure float32 for sounddevice)
        outdata[:, 0] = mixed.astype(np.float32)
        outdata[:, 1] = mixed.astype(np.float32)
        
        # Update time
        self.current_time += frames / self.sample_rate
        
        # Check if we've passed the end of the loop/arcs
        loop_end = self.page.settings.loop_end if self.page.settings.loop_enabled else self.page.total_duration
        if loop_end > 0 and self.current_time > loop_end:
            if self.page.settings.loop_enabled:
                # Reset all voices for clean loop
                for voice in self.voices:
                    voice.reset()
                self.current_time = self.page.settings.loop_start
                self._assign_voices()
            else:
                self.playing = False
        
        # Notify time update
        if self.on_time_update:
            self.on_time_update(self.current_time)
    
    def render_to_array(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        stereo: bool = True
    ) -> NDArray[np.float64]:
        """
        Render audio to a numpy array (offline rendering).
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds (default: end of page)
            stereo: If True, return stereo array
            
        Returns:
            Audio samples as numpy array
        """
        if self.page is None:
            return np.array([])
        
        if end_time is None:
            end_time = self.page.total_duration
        
        duration = end_time - start_time
        total_samples = int(duration * self.sample_rate)
        
        # Assign voices
        self.current_time = start_time
        self._assign_voices()
        
        # Render in chunks
        output = np.zeros(total_samples)
        samples_rendered = 0
        
        while samples_rendered < total_samples:
            chunk_size = min(self.buffer_size, total_samples - samples_rendered)
            
            # Mix all voices
            mixed = np.zeros(chunk_size)
            active_count = 0
            
            for voice in self.voices:
                if voice.active:
                    samples = voice.render(
                        self.current_time,
                        chunk_size,
                        self.sample_rate
                    )
                    mixed += samples
                    active_count += 1
            
            # Normalize
            if active_count > 0:
                mixed = mixed / max(active_count, 1)
            
            mixed = mixed * self.master_volume

            # Apply same post-FX chain as real-time playback
            mixed = self._apply_effects(mixed.astype(np.float64))

            # Match real-time final output stage
            mixed = self._apply_output_limiter(mixed)
            
            output[samples_rendered:samples_rendered + chunk_size] = mixed
            
            samples_rendered += chunk_size
            self.current_time += chunk_size / self.sample_rate
        
        if stereo:
            return np.column_stack([output, output])
        return output
    
    def render_to_file(
        self,
        filepath: str,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> None:
        """
        Render audio to a WAV file.
        
        Args:
            filepath: Output file path
            start_time: Start time in seconds
            end_time: End time in seconds
        """
        from scipy.io import wavfile
        
        audio = self.render_to_array(start_time, end_time, stereo=True)
        
        # Convert to 16-bit integer
        audio_int = (audio * 32767).astype(np.int16)
        
        wavfile.write(filepath, self.sample_rate, audio_int)


def preview_waveform(waveform: Waveform, duration: float = 1.0, frequency: float = 440.0) -> None:
    """
    Preview a waveform by playing it at a fixed frequency.
    
    Args:
        waveform: Waveform to preview
        duration: Duration in seconds
        frequency: Frequency in Hz
    """
    if not AUDIO_AVAILABLE:
        print("Audio not available")
        return
    
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    # Generate phase array
    phase_increments = frequency / sample_rate
    phases = np.cumsum(np.full(num_samples, phase_increments)) % 1.0
    
    # Get samples
    samples = waveform.get_samples_at_phases(phases)
    
    # Apply simple fade in/out
    fade_samples = int(0.01 * sample_rate)
    samples[:fade_samples] *= np.linspace(0, 1, fade_samples)
    samples[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Play
    sd.play(samples * 0.5, sample_rate)
    sd.wait()


def preview_envelope(envelope: Envelope, duration: float = 1.0, frequency: float = 440.0) -> None:
    """
    Preview an envelope by applying it to a sine wave.
    
    Args:
        envelope: Envelope to preview
        duration: Duration in seconds
        frequency: Frequency in Hz
    """
    if not AUDIO_AVAILABLE:
        print("Audio not available")
        return
    
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    
    # Generate sine wave
    t = np.linspace(0, duration, num_samples)
    samples = np.sin(2 * np.pi * frequency * t)
    
    # Apply envelope
    positions = np.linspace(0, 1, num_samples)
    amplitudes = envelope.get_amplitudes_at_positions(positions)
    samples = samples * amplitudes
    
    # Play
    sd.play(samples * 0.5, sample_rate)
    sd.wait()

