"""
Audio pipeline preview (no BCI experiment required).

Plays a short generated arc through the SAME synthesis + FX chain used by the app:
- FrequencyTable mapping (e.g., "Hackathon Safe")
- Voice FM (optional)
- Synthesizer post-FX (EQ/phaser/distortion)
- Final soft-clip limiter

Usage examples:
  python preview_audio_pipeline.py
  python preview_audio_pipeline.py --duration 6 --waveform Sawtooth
  python preview_audio_pipeline.py --phaser --phaser-mix 0.35
  python preview_audio_pipeline.py --distortion --drive 2.5 --distortion-mix 0.25
  python preview_audio_pipeline.py --fm --mod-index 2.0
  python preview_audio_pipeline.py --mode realtime
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import numpy as np


def _build_preview_page(
    *,
    duration: float,
    waveform: str,
    envelope: str,
    amplitude: float,
    frequency_table_name: str,
    fm: bool,
    mod_index: float,
) -> "Page":
    from src.core.page import Page
    from src.core.arc import Arc

    page = Page(name="Audio Preview")
    page.settings.duration = float(duration)
    page.settings.loop_end = float(duration)
    page.settings.loop_enabled = False
    page.settings.frequency_table_name = frequency_table_name

    # Carrier arc: sweeps pitch up then down across most of the range.
    carrier = Arc(
        name="Preview Sweep",
        waveform_name=waveform,
        envelope_name=envelope,
        amplitude=float(amplitude),
        pan=0.0,
    )

    n_points = 240
    half = n_points // 2
    times = np.linspace(0.0, duration, n_points)
    pitches = np.concatenate(
        [
            np.linspace(0.08, 0.92, half, endpoint=False),
            np.linspace(0.92, 0.08, n_points - half),
        ]
    )
    for t, p in zip(times.tolist(), pitches.tolist()):
        carrier.add_point(float(t), float(p))

    page.add_arc(carrier)

    if fm:
        # Modulator arc: constant pitch, muted, used only for FM.
        modulator = Arc(
            name="FM Modulator (muted)",
            waveform_name="Sine",
            envelope_name=envelope,
            amplitude=1.0,
            pan=0.0,
            muted=True,
        )

        # Modulator pitch near upper-mid to create audible FM movement.
        for t in np.linspace(0.0, duration, 60).tolist():
            modulator.add_point(float(t), 0.75)

        page.add_arc(modulator)

        carrier.modulator_id = modulator.id
        carrier.modulation_index = float(mod_index)

    return page


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview synth + FX audio pipeline.")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--waveform", type=str, default="Sine", choices=["Sine", "Triangle", "Sawtooth", "Square"])
    parser.add_argument("--envelope", type=str, default="ADSR")
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument("--volume", type=float, default=0.5, help="Synth master volume (0-1).")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--table", type=str, default="Hackathon Safe")

    parser.add_argument("--mode", type=str, default="offline", choices=["offline", "realtime"])
    parser.add_argument(
        "--wav-out",
        type=str,
        default=None,
        help="Optional path to write a WAV file (useful if playback libs missing).",
    )

    # FX toggles/params
    parser.add_argument("--no-fx", action="store_true", help="Disable Synthesizer effects chain.")
    parser.add_argument("--hp", type=float, default=30.0, help="High-pass cutoff (Hz).")
    parser.add_argument("--lp", type=float, default=16000.0, help="Low-pass cutoff (Hz).")

    parser.add_argument("--phaser", action="store_true")
    parser.add_argument("--phaser-mix", type=float, default=None, help="0=dry, 1=wet (default becomes audible when --phaser).")
    parser.add_argument("--phaser-rate", type=float, default=0.25)
    parser.add_argument("--phaser-min", type=float, default=300.0)
    parser.add_argument("--phaser-max", type=float, default=1500.0)
    parser.add_argument("--phaser-stages", type=int, default=4)
    parser.add_argument("--phaser-feedback", type=float, default=0.0)

    parser.add_argument("--distortion", action="store_true")
    parser.add_argument("--drive", type=float, default=None, help="Distortion drive (default becomes audible when --distortion).")
    parser.add_argument("--distortion-mix", type=float, default=None, help="0=dry, 1=wet (default becomes audible when --distortion).")
    parser.add_argument("--distortion-mode", type=str, default=None, choices=["tanh", "hardclip", "guitar"])
    parser.add_argument("--distortion-clip", type=float, default=None, help="Hardclip threshold (smaller = more extreme).")
    parser.add_argument("--distortion-os", type=int, default=None, choices=[1, 2, 4], help="Oversampling factor for distortion (reduces harsh digital clipping).")
    parser.add_argument("--bass-db", type=float, default=None, help="Low-shelf bass boost in dB (requires synth support).")
    parser.add_argument("--bass-hz", type=float, default=120.0, help="Bass shelf frequency (Hz).")
    parser.add_argument("--fuzz-lp", type=float, default=None, help="Post-distortion low-pass (Hz) for fuzz tone.")

    # FM toggle
    parser.add_argument("--fm", action="store_true", help="Enable FM (adds muted modulator arc).")
    parser.add_argument("--mod-index", type=float, default=2.0)

    args = parser.parse_args()

    from src.core.synthesizer import Synthesizer

    # Build page
    page = _build_preview_page(
        duration=args.duration,
        waveform=args.waveform,
        envelope=args.envelope,
        amplitude=args.amplitude,
        frequency_table_name=args.table,
        fm=args.fm,
        mod_index=args.mod_index,
    )

    # Configure synth
    synth = Synthesizer(sample_rate=int(args.sample_rate))
    synth.set_page(page)
    synth.master_volume = float(np.clip(args.volume, 0.0, 1.0))

    # Configure effects (these are applied before playback/output)
    synth.effects.enabled = not args.no_fx
    synth.effects.highpass_hz = float(args.hp)
    synth.effects.lowpass_hz = float(args.lp)

    synth.effects.phaser_enabled = bool(args.phaser)
    if args.phaser:
        # If user didn't specify, pick an audible default.
        synth.effects.phaser_mix = float(0.6 if args.phaser_mix is None else args.phaser_mix)
    else:
        synth.effects.phaser_mix = float(0.0 if args.phaser_mix is None else args.phaser_mix)
    synth.effects.phaser_rate_hz = float(args.phaser_rate)
    synth.effects.phaser_min_hz = float(args.phaser_min)
    synth.effects.phaser_max_hz = float(args.phaser_max)
    synth.effects.phaser_stages = int(args.phaser_stages)
    synth.effects.phaser_feedback = float(args.phaser_feedback)

    synth.effects.distortion_enabled = bool(args.distortion)
    if args.distortion:
        # Fuzzier default voicing
        # Make it VISIBLY distorted in time-domain (flat tops) by default.
        synth.effects.distortion_drive = float(50.0 if args.drive is None else args.drive)
        synth.effects.distortion_mix = float(1.0 if args.distortion_mix is None else args.distortion_mix)
        synth.effects.distortion_mode = str("hardclip" if args.distortion_mode is None else args.distortion_mode)
        synth.effects.distortion_clip = float(0.02 if args.distortion_clip is None else args.distortion_clip)
        synth.effects.distortion_oversample = int(4 if args.distortion_os is None else args.distortion_os)

        # Bassier: add low-shelf boost + slightly darker post-LP to keep it fuzzy (not harsh)
        # Also relax HP slightly if user left it at the default.
        if float(args.hp) == 30.0:
            synth.effects.highpass_hz = 5.0
        # Keep more top-end so the hard clip is obvious in waveform
        if float(args.lp) == 16000.0:
            synth.effects.lowpass_hz = 16000.0
        synth.effects.bass_boost_enabled = True
        synth.effects.bass_boost_db = float(18.0 if args.bass_db is None else args.bass_db)
        synth.effects.bass_boost_hz = float(args.bass_hz)
        synth.effects.fuzz_tone_lowpass_hz = float(8000.0 if args.fuzz_lp is None else args.fuzz_lp)
    else:
        synth.effects.distortion_drive = float(1.0 if args.drive is None else args.drive)
        synth.effects.distortion_mix = float(0.0 if args.distortion_mix is None else args.distortion_mix)
        synth.effects.distortion_mode = str("tanh" if args.distortion_mode is None else args.distortion_mode)
        synth.effects.distortion_clip = float(0.25 if args.distortion_clip is None else args.distortion_clip)
        synth.effects.distortion_oversample = int(1 if args.distortion_os is None else args.distortion_os)
        synth.effects.bass_boost_enabled = False
        synth.effects.bass_boost_db = 0.0
        synth.effects.fuzz_tone_lowpass_hz = 0.0

    # Re-init effect state arrays if stage count changed
    synth.reset_effects_state()

    # Print a small config summary for troubleshooting
    print(f"[preview] duration={args.duration}s, sr={synth.sample_rate}, table='{args.table}', waveform={args.waveform}")
    print(f"[preview] fx={not args.no_fx}, phaser={args.phaser}, distortion={args.distortion}, fm={args.fm}")
    try:
        print(f"[preview] effects={asdict(synth.effects)}")
    except Exception:
        pass

    # Play
    if args.mode == "realtime":
        # Uses Synthesizer._audio_callback (real-time path)
        synth.play(0.0)
        # Wait until playback completes, with a safety timeout
        t0 = time.time()
        while synth.playing and (time.time() - t0) < (args.duration + 2.0):
            time.sleep(0.05)
        synth.stop()
        return 0

    # Offline render path (used by BCI play_score -> synthesize_score -> render_to_array)
    audio = synth.render_to_array(0.0, float(args.duration), stereo=True)
    if audio is None or len(audio) == 0:
        print("[preview] no audio generated")
        return 1

    # Try direct playback via sounddevice; if unavailable, fall back to WAV + winsound (Windows)
    audio_f32 = audio.astype(np.float32)

    if args.wav_out:
        out_path = args.wav_out
    else:
        out_path = None

    try:
        import sounddevice as sd  # type: ignore
        sd.play(audio_f32, samplerate=synth.sample_rate)
        sd.wait()
        # Optionally still write a WAV for inspection
        if out_path:
            _write_wav(out_path, audio_f32, synth.sample_rate)
        return 0
    except Exception as e:
        print(f"[preview] sounddevice playback unavailable: {e}")

    # Fallback: write WAV and try winsound
    if out_path is None:
        import tempfile
        out_path = tempfile.mkstemp(prefix="upic_preview_", suffix=".wav")[1]

    _write_wav(out_path, audio_f32, synth.sample_rate)
    print(f"[preview] wrote WAV: {out_path}")

    try:
        import winsound  # Windows only
        winsound.PlaySound(out_path, winsound.SND_FILENAME)
        return 0
    except Exception as e:
        print(f"[preview] winsound playback unavailable: {e}")
        print("[preview] Please play the WAV file manually.")
        return 0
    return 0


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write stereo float32 audio [-1,1] to 16-bit PCM WAV."""
    import wave

    a = audio
    if a.ndim == 1:
        a = np.column_stack([a, a])
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"Expected stereo audio (n,2), got shape={a.shape}")

    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())

