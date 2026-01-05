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
    
    # Callbacks
    on_time_update: Optional[Callable[[float], None]] = None
    on_playback_stop: Optional[Callable[[], None]] = None
    
    def __post_init__(self) -> None:
        """Initialize voices."""
        self.voices = [Voice() for _ in range(self.num_voices)]
    
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
        
        # Soft clip to prevent harsh distortion
        mixed = np.tanh(mixed * 0.8) * 0.9
        
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
            mixed = np.clip(mixed, -1, 1)
            
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

