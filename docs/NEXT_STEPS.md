# UPIC Clone - Next Steps

**Created:** 2026-01-05  
**Last Updated:** 2026-01-05

---

## 1. Visual Distinction for Carrier vs Modulator Arcs

**Priority:** High  
**Status:** Planned

### Current Behavior
- All arcs are drawn in the same blue color
- Muted arcs are gray/dotted
- Orange dashed arrows show modulation connections

### Desired Behavior
- **Carrier arcs**: Distinct color (e.g., blue/cyan)
- **Modulator arcs**: Different color (e.g., orange/yellow)
- Arcs that are BOTH carrier and modulator: Third color or gradient
- Color should update dynamically when links are created/removed

### Implementation Ideas
- Add `is_modulator` property to Arc (check if any arc uses this as modulator)
- Add `is_carrier` property to Arc (check if this arc has a modulator_id)
- Update `_draw_arc()` in `page_canvas.py` to use different colors based on role
- Consider color legend in UI

---

## 2. Real-Time Envelope/Parameter Changes

**Priority:** High  
**Status:** Planned

### Current Behavior
- Changes to arc properties take effect, but you have to wait for the loop to restart
- No immediate audible feedback when adjusting sliders

### Desired Behavior
- Envelope changes apply immediately to playing arcs
- Amplitude/pan changes heard in real-time
- Waveform changes apply on next cycle (to avoid clicks)

### Implementation Ideas
- The synthesizer already re-reads arc properties each frame via `_assign_voices()`
- Issue may be that voices cache waveform/envelope objects at `start()`
- Solution: Have voices reference arc properties directly, or refresh on each render
- For envelope: Could interpolate between old/new to avoid clicks

### Technical Approach
```python
# In Voice.render(), instead of using cached self.envelope:
envelope = self.arc.get_current_envelope()  # Fetch fresh each time
# Or have synthesizer update voice.envelope when arc changes
```

---

## 3. Image-to-Score Conversion

**Priority:** Medium  
**Status:** Exploration

### Concept
- Load a black & white (or grayscale) image
- Convert to arcs that can be played as music
- Black pixels = sound, white pixels = silence (or vice versa)
- Vertical axis = pitch, horizontal axis = time

### Implementation Ideas

#### Option A: Contour Tracing
- Use edge detection (Canny, Sobel) to find contours
- Convert contours to arc paths
- Each contour becomes one arc
- Pros: Clean arcs, musical results
- Cons: Loses filled regions

#### Option B: Horizontal Scanlines
- For each row (pitch level), scan horizontally
- Create arc segments where black pixels exist
- Pros: Preserves all image data
- Cons: May create many small arcs

#### Option C: Threshold + Connected Components
- Threshold image to binary
- Find connected black regions
- Trace outline of each region as an arc
- Pros: Good balance
- Cons: Complex shapes may not sound musical

### Dependencies
- `PIL` / `Pillow` for image loading
- `opencv-python` for advanced processing (optional)
- `scikit-image` for contour detection (optional)

### UI Flow
1. File → Import Image...
2. Dialog shows image preview with options:
   - Threshold slider (for grayscale → binary)
   - Invert checkbox
   - Time duration mapping
   - Pitch range mapping
3. Preview arcs overlaid on image
4. Import button creates arcs on page

### Code Structure
```
src/
  utils/
    image_import.py  # Image processing functions
  gui/
    image_import_dialog.py  # Import dialog UI
```

---

## 4. Scale/Key Locking (Pitch Quantization)

**Priority:** High  
**Status:** Planned

### Concept
Lock the pitch axis to specific musical scales so all drawn notes are "in key."
No wrong notes - everything harmonizes automatically.

### Desired Behavior
- Dropdown to select scale: C Major, A Minor, D Dorian, Pentatonic, Blues, etc.
- Dropdown to select root note: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- Drawn pitches snap to nearest note in the selected scale
- Option for "snap while drawing" vs "snap on playback"
- Visual: Grid lines show valid pitches

### Scale Definitions
```python
SCALES = {
    "Major":        [0, 2, 4, 5, 7, 9, 11],  # W W H W W W H
    "Minor":        [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    "Harmonic Minor": [0, 2, 3, 5, 7, 8, 11],
    "Melodic Minor": [0, 2, 3, 5, 7, 9, 11],
    "Dorian":       [0, 2, 3, 5, 7, 9, 10],
    "Phrygian":     [0, 1, 3, 5, 7, 8, 10],
    "Lydian":       [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian":   [0, 2, 4, 5, 7, 9, 10],
    "Locrian":      [0, 1, 3, 5, 6, 8, 10],
    "Pentatonic Major": [0, 2, 4, 7, 9],
    "Pentatonic Minor": [0, 3, 5, 7, 10],
    "Blues":        [0, 3, 5, 6, 7, 10],
    "Chromatic":    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # All notes
    "Whole Tone":   [0, 2, 4, 6, 8, 10],
}
```

### Implementation
1. Extend `FrequencyTable` to support scale-locked modes
2. Add `scale_name` and `root_note` to `PageSettings`
3. Create `FrequencyTable.from_scale(scale_name, root_note, octave_range)`
4. UI: Add scale/key dropdowns to toolbar or settings panel
5. Update grid drawing to show scale degrees

### UI Mockup
```
[Scale: Major ▼] [Key: A ▼] [Octaves: 4 ▼]
```

---

## 5. Time Quantization (Rhythmic Grid)

**Priority:** High  
**Status:** Planned

### Concept
Quantize note timing to a rhythmic grid for more structured, rhythmic music.
Like a step sequencer overlay on the freeform canvas.

### Desired Behavior
- Toggle: Quantize Mode ON/OFF
- Grid resolution: 1/4, 1/8, 1/16, 1/32 notes (or seconds/ms)
- BPM setting for musical timing
- Options:
  - **Snap Start**: Arc start times snap to grid
  - **Snap Duration**: Arc lengths snap to grid divisions
  - **Gate Mode**: Sound only plays at grid intervals (staccato effect)

### Gate Mode Detail
Instead of continuous sound following the arc, sound triggers at each grid point:
- Arc defines pitch trajectory
- Sound "pulses" at each 1/8 note (or selected division)
- Creates rhythmic, sequencer-like patterns
- Amplitude envelope restarts at each gate

### Visual Feedback
- Vertical grid lines at beat divisions
- Beat 1 = thicker line
- Show BPM in transport bar
- Optional: Beat counter display

### Implementation
```python
@dataclass
class RhythmSettings:
    enabled: bool = False
    bpm: float = 120.0
    division: int = 8  # 1/8 notes
    gate_mode: bool = False
    gate_length: float = 0.5  # 50% of division
    swing: float = 0.0  # -1 to 1, shuffle feel

# In synthesizer, if gate_mode:
def apply_gate(samples, current_time, rhythm_settings):
    beat_duration = 60.0 / rhythm_settings.bpm
    division_duration = beat_duration / (rhythm_settings.division / 4)
    
    # Calculate gate envelope
    position_in_division = current_time % division_duration
    gate_time = division_duration * rhythm_settings.gate_length
    
    if position_in_division > gate_time:
        return samples * 0  # Silence between gates
    return samples
```

### UI Mockup
```
[⏱ Quantize] [BPM: 120] [1/8 ▼] [☑ Gate Mode]
```

---

## 6. Expressiveness Enhancements

**Priority:** Medium  
**Status:** Ideas

### Velocity/Dynamics
- Draw with pressure sensitivity (if tablet)
- Thicker line = louder
- Or: Vertical position within arc = velocity

### Articulation Modes
- **Legato**: Smooth pitch transitions (current behavior)
- **Portamento**: Glide between pitches with adjustable time
- **Staccato**: Short, detached notes (via gate mode)

### Per-Arc Effects
- Vibrato depth/rate
- Tremolo depth/rate  
- Pitch bend range
- Filter cutoff (if we add filters)

### Global Effects (Future)
- Reverb
- Delay
- Chorus
- Filter with envelope

### Microtuning
- Support for non-12-TET tunings
- Just intonation
- Custom cent offsets per note
- Import Scala (.scl) files

---

## 7. Other Ideas for Future

### Quick Wins
- Keyboard shortcuts for common actions (mute = M, solo = S)
- Copy/paste arcs (Ctrl+C, Ctrl+V)
- Duplicate arc (Ctrl+D)
- Snap to grid option

### Medium Effort
- Multiple pages with tabs
- Frequency table editor (custom scales)
- Audio export progress bar
- Recent files menu

### Larger Features
- MIDI input (draw with MIDI controller)
- OSC support for external control
- Spectral display mode
- Recording input audio as arcs

---

## Technical Debt / Cleanup

- Remove debug print statements from synthesizer
- Add unit tests for core modules
- Document public APIs with docstrings
- Create user manual / tutorial

---

## Session Notes

### 2026-01-05 Session Summary
- Implemented full FM modulation with Ctrl+Click linking
- Added Arc Properties Panel
- Fixed recursion bug in modulator chains
- Fixed audio clipping and oscillation issues
- Added loop playback
- Added Clear All button
- Core UPIC functionality is now working

### What's Working Well
- Drawing arcs ✓
- Waveform/envelope selection ✓
- Real-time playback ✓
- FM modulation ✓
- Loop mode ✓
- Save/load projects ✓

### Known Issues
- Envelope changes don't apply until loop restarts
- No visual distinction between carrier/modulator arcs (just arrows)
- Image import not yet implemented

