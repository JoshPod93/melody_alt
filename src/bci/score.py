"""
BCI Score module for BCI-UPIC.

Handles conversion of BCI cursor trails into musical scores
that can be played back through the existing synthesis system.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Import from existing core modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.arc import Arc, ArcPoint
from src.core.page import Page, PageSettings


@dataclass
class BCIScore:
    """
    A musical score generated from BCI cursor movement.
    
    Converts the trail of cursor positions into an Arc that can
    be synthesized using the existing UPIC synthesis engine.
    
    Attributes:
        trail: List of (time, pitch) points from BCI session
        duration: Total duration in seconds
        waveform_name: Waveform to use for synthesis
        envelope_name: Envelope to use for synthesis
        amplitude: Base amplitude (0-1)
        created_at: Timestamp of creation
        metadata: Additional metadata about the BCI session
    """
    trail: List[Tuple[float, float]] = field(default_factory=list)
    duration: float = 10.0
    waveform_name: str = "Sine"
    envelope_name: str = "ADSR"
    amplitude: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Derived arc
    _arc: Optional[Arc] = field(default=None, repr=False)
    
    def set_trail(self, trail: List[Tuple[float, float]]) -> None:
        """Set the trail from cursor positions."""
        self.trail = trail
        self._arc = None  # Invalidate cached arc
    
    def to_arc(self, name: str = "BCI Arc") -> Arc:
        """
        Convert the trail to an Arc for synthesis.
        
        Args:
            name: Name for the arc
            
        Returns:
            Arc object ready for synthesis
        """
        if self._arc is not None:
            return self._arc
        
        arc = Arc(
            name=name,
            waveform_name=self.waveform_name,
            envelope_name=self.envelope_name,
            amplitude=self.amplitude,
            color=(66, 200, 135)  # Green for BCI-generated
        )
        
        # Convert trail to arc points
        for time, pitch in self.trail:
            arc.add_point(time, pitch)
        
        self._arc = arc
        return arc
    
    def to_page(self, name: str = "BCI Composition") -> Page:
        """
        Convert the score to a Page for synthesis.
        
        Args:
            name: Name for the page
            
        Returns:
            Page object ready for synthesis
        """
        page = Page(name=name)
        page.settings.duration = self.duration
        page.settings.loop_end = self.duration
        # Ensure BCI playback uses the safe, narrowed pitch range by default.
        page.settings.frequency_table_name = "Hackathon Safe"
        
        arc = self.to_arc(f"{name} Arc")
        page.add_arc(arc)
        
        return page
    
    def simplify_trail(self, tolerance: float = 0.01) -> List[Tuple[float, float]]:
        """
        Simplify the trail using Douglas-Peucker algorithm.
        
        Reduces the number of points while preserving the shape.
        
        Args:
            tolerance: Maximum distance from original path
            
        Returns:
            Simplified trail
        """
        if len(self.trail) < 3:
            return self.trail.copy()
        
        points = np.array(self.trail)
        
        # Normalize to similar scales
        time_scale = self.duration
        points_normalized = points.copy()
        points_normalized[:, 0] /= time_scale
        
        # Douglas-Peucker simplification
        simplified_indices = self._douglas_peucker(points_normalized, tolerance)
        
        return [self.trail[i] for i in simplified_indices]
    
    def _douglas_peucker(
        self,
        points: np.ndarray,
        tolerance: float
    ) -> List[int]:
        """
        Douglas-Peucker line simplification algorithm.
        
        Args:
            points: Array of (x, y) points
            tolerance: Maximum perpendicular distance
            
        Returns:
            List of indices to keep
        """
        if len(points) <= 2:
            return list(range(len(points)))
        
        # Find point with maximum distance from line
        start = points[0]
        end = points[-1]
        
        # Line from start to end
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return [0, len(points) - 1]
        
        line_unit = line_vec / line_len
        
        # Calculate perpendicular distances
        max_dist = 0
        max_idx = 0
        
        for i in range(1, len(points) - 1):
            point_vec = points[i] - start
            proj_len = np.dot(point_vec, line_unit)
            proj_point = start + proj_len * line_unit
            dist = np.linalg.norm(points[i] - proj_point)
            
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_dist > tolerance:
            left = self._douglas_peucker(points[:max_idx + 1], tolerance)
            right = self._douglas_peucker(points[max_idx:], tolerance)
            
            # Combine results (avoiding duplicate middle point)
            return left[:-1] + [i + max_idx for i in right]
        else:
            return [0, len(points) - 1]
    
    def to_dict(self) -> dict:
        """Serialize score to dictionary."""
        return {
            'trail': self.trail,
            'duration': self.duration,
            'waveform_name': self.waveform_name,
            'envelope_name': self.envelope_name,
            'amplitude': self.amplitude,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> BCIScore:
        """Deserialize score from dictionary."""
        score = cls(
            trail=data.get('trail', []),
            duration=data.get('duration', 10.0),
            waveform_name=data.get('waveform_name', 'Sine'),
            envelope_name=data.get('envelope_name', 'ADSR'),
            amplitude=data.get('amplitude', 0.5),
            metadata=data.get('metadata', {})
        )
        
        if 'created_at' in data:
            score.created_at = datetime.fromisoformat(data['created_at'])
        
        return score
    
    def save(self, filepath: Path | str) -> None:
        """Save score to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path | str) -> BCIScore:
        """Load score from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics about the score.
        
        Returns:
            Dictionary of statistics
        """
        if not self.trail:
            return {}
        
        pitches = [p[1] for p in self.trail]
        times = [p[0] for p in self.trail]
        
        # Calculate pitch changes
        pitch_changes = np.diff(pitches)
        
        return {
            'duration': self.duration,
            'num_points': len(self.trail),
            'pitch_mean': np.mean(pitches),
            'pitch_std': np.std(pitches),
            'pitch_min': np.min(pitches),
            'pitch_max': np.max(pitches),
            'pitch_range': np.max(pitches) - np.min(pitches),
            'total_movement': np.sum(np.abs(pitch_changes)),
            'direction_changes': np.sum(np.diff(np.sign(pitch_changes)) != 0)
        }


def synthesize_score(
    score: BCIScore,
    output_path: Optional[Path | str] = None
) -> Optional[np.ndarray]:
    """
    Synthesize a BCI score to audio.
    
    Args:
        score: BCIScore to synthesize
        output_path: Optional path to save WAV file
        
    Returns:
        Audio samples as numpy array, or None if saved to file
    """
    from src.core.synthesizer import Synthesizer
    
    # Create page from score
    page = score.to_page()
    
    # Create synthesizer
    synth = Synthesizer()
    synth.set_page(page)

    # Hackathon preset: bassier, fuzzier playback (no phaser).
    # This affects BOTH manual Play and auto-play-after-capture since they share this path.
    synth.effects.enabled = True
    synth.effects.phaser_enabled = False
    synth.effects.phaser_mix = 0.0

    # Allow more low-end through, then boost bass.
    synth.effects.highpass_hz = 5.0
    synth.effects.lowpass_hz = 12000.0

    # Fuzz voicing
    synth.effects.distortion_enabled = True
    synth.effects.distortion_drive = 50.0
    synth.effects.distortion_mix = 1.0
    synth.effects.distortion_mode = "hardclip"
    synth.effects.distortion_clip = 0.02
    synth.effects.distortion_oversample = 4
    synth.effects.bass_boost_enabled = True
    synth.effects.bass_boost_db = 18.0
    synth.effects.bass_boost_hz = 90.0
    synth.effects.bass_boost_slope = 1.0
    synth.effects.fuzz_tone_lowpass_hz = 8000.0

    # Prevent "bad clipping" at the very end by linearly scaling under a ceiling
    synth.effects.auto_level = True
    synth.effects.output_ceiling = 0.9

    synth.reset_effects_state()
    
    # Render to array
    audio = synth.render_to_array(0, score.duration, stereo=True)
    
    # Save if path provided
    if output_path:
        synth.render_to_file(str(output_path), 0, score.duration)
        return None
    
    return audio


def play_score(score: BCIScore) -> None:
    """
    Play a BCI score through the audio system.
    
    Args:
        score: BCIScore to play
    """
    if not score.trail or len(score.trail) < 2:
        print("Warning: Score has no trail data")
        return
    
    try:
        # Synthesize to array first (more reliable)
        audio = synthesize_score(score)
        
        if audio is not None and len(audio) > 0:
            # audio shape is (2, n_samples) for stereo - need (n_samples, 2)
            if audio.ndim == 2:
                if audio.shape[0] == 2:
                    audio = audio.T  # Transpose from (2, N) to (N, 2)
                # If shape[1] == 2, it's already correct
            elif audio.ndim == 1:
                # Mono - make stereo
                audio = np.column_stack([audio, audio])
            
            # Ensure float32 and normalize
            audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))

            # Prefer sounddevice; fall back to Windows wav playback if missing.
            try:
                import sounddevice as sd
                sd.play(audio, samplerate=44100)
                sd.wait()
            except Exception as e:
                print(f"Playback warning: sounddevice unavailable ({e}); using WAV fallback.")
                _play_via_wav_fallback(audio, sample_rate=44100)
        else:
            print("Warning: No audio generated")
    except Exception as e:
        print(f"Playback error: {e}")
        raise


def _play_via_wav_fallback(audio: np.ndarray, sample_rate: int = 44100) -> None:
    """
    Fallback playback for environments without sounddevice.
    Writes a temporary WAV and plays it via winsound on Windows.
    """
    import tempfile
    import wave

    a = audio
    if a.ndim == 1:
        a = np.column_stack([a, a])
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"Expected stereo audio (n,2), got shape={a.shape}")

    a = np.clip(a, -1.0, 1.0).astype(np.float32)
    pcm = (a * 32767.0).astype(np.int16)

    wav_path = tempfile.mkstemp(prefix="bci_score_", suffix=".wav")[1]
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())

    try:
        import winsound
        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"Playback fallback failed: {e}. WAV written to: {wav_path}")


if __name__ == "__main__":
    # Test score creation and synthesis
    print("Testing BCI score system...")
    
    # Create a test trail (sine wave pattern)
    duration = 5.0
    num_points = 100
    times = np.linspace(0, duration, num_points)
    pitches = 0.5 + 0.3 * np.sin(2 * np.pi * 0.5 * times)  # 0.5 Hz sine wave
    
    trail = list(zip(times.tolist(), pitches.tolist()))
    
    # Create score
    score = BCIScore(
        trail=trail,
        duration=duration,
        waveform_name="Sine",
        metadata={'test': True}
    )
    
    # Get statistics
    stats = score.get_statistics()
    print(f"Score statistics: {stats}")
    
    # Simplify trail
    simplified = score.simplify_trail(tolerance=0.02)
    print(f"Original points: {len(trail)}, Simplified: {len(simplified)}")
    
    # Convert to page
    page = score.to_page()
    print(f"Page created with {len(page.arcs)} arc(s)")
