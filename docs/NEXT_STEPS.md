# UPIC Clone - Next Steps

**Created:** 2026-01-05  
**Last Updated:** 2026-01-05

---

## ⚠️ CRITICAL DESIGN CONTEXT: BCI Accessibility

**This project is intended for Brain-Computer Interface (BCI) music users.**

### Core Principle
Not all users will have high-resolution cursor control. Some BCI control signals may be:
- **Low spatial resolution** (can't draw precise curves)
- **Low temporal resolution** (slow updates, latency)
- **Binary or discrete** (select from options, not continuous movement)
- **Single-switch** (one signal, like a blink or thought)
- **Multi-class but limited** (e.g., 4 directions + select)

### Design Implications

**Every feature must consider:**
1. Can this work with a 4-direction joystick + select button?
2. Can this work with single-switch scanning?
3. Can this work with eye gaze (dwell-click)?
4. Can this work with low-resolution head tracking?
5. Is there a "coarse" alternative to fine motor actions?

**Avoid requiring:**
- Precise cursor positioning
- Fast, continuous mouse movement
- Click-and-drag (hard for many BCI users)
- Small click targets
- Time-pressure interactions

**Prefer:**
- Large click/selection targets
- Discrete choices (menus, palettes)
- Step-by-step workflows
- Undo-friendly actions
- Auto-assist features (snap, quantize, templates)

### Input Abstraction Layer (Future)
Create an abstraction so the app can receive input from:
- Mouse/trackpad (current)
- Keyboard-only navigation
- Gamepad/joystick
- Eye tracker (Tobii, etc.)
- BCI systems (OpenBCI, Emotiv, etc.)
- OSC/MIDI control signals
- Custom accessibility switches

```python
class InputAdapter:
    """Abstract input source for BCI compatibility."""
    def get_position(self) -> Tuple[float, float]: ...
    def is_selecting(self) -> bool: ...
    def get_discrete_command(self) -> Optional[str]: ...  # "up", "down", "select", etc.
```

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

## 7. VST/Plugin Host Integration

**Priority:** Medium  
**Status:** Exploration

### Concept
Route audio through VST/AU plugins for effects processing (reverb, delay, filters, etc.)

### Options

#### Option A: External DAW Routing
- Output audio via virtual audio cable (VB-Cable, BlackHole, JACK)
- User loads effects in their DAW
- Pros: Full plugin compatibility, user's existing setup
- Cons: Requires external software setup

#### Option B: Embedded Plugin Host
- Use `pedalboard` library (Spotify's Python VST host)
- Load VST3/AU plugins directly in app
- Pros: Self-contained, integrated
- Cons: Complex, potential compatibility issues

#### Option C: Built-in Effects
- Implement common effects natively (reverb, delay, filter, distortion)
- No external dependencies
- Pros: Simple, portable, BCI-friendly (no complex plugin UIs)
- Cons: Limited to what we build

### Recommended Approach
1. Start with **Option A** (virtual audio routing) - works now
2. Add **Option C** (built-in effects) - accessible UI
3. Consider **Option B** later for power users

### BCI Consideration
Plugin UIs are often inaccessible. Built-in effects with large, simple controls are better for BCI users.

---

## 8. BCI-Friendly Alternative Input Modes

**Priority:** CRITICAL  
**Status:** Planned

### Problem
Current interface requires:
- Click and drag to draw (hard for BCI)
- Precise cursor positioning
- Continuous mouse movement

### Solution: Multiple Drawing Modes

#### Mode A: Template/Stamp Mode
- Pre-made arc shapes (line, curve, zigzag, wave)
- Click to place, resize with simple controls
- BCI-friendly: Select template → Place → Adjust

```
[Templates]
[━━━] Horizontal line
[╱╲╱] Zigzag
[∿∿∿] Sine wave  
[╱━━] Ramp up
[━━╲] Ramp down
[⌒⌒] Arc/curve
```

#### Mode B: Point-by-Point Mode
- Click to place points, system connects them
- No dragging required
- Adjustable interpolation (linear, smooth, stepped)
- BCI-friendly: Click → Click → Click → Done

#### Mode C: Grid/Step Sequencer Mode
- Divide canvas into large clickable cells
- Click cell to toggle note on/off
- Like a piano roll with big buttons
- BCI-friendly: Large targets, no precision needed

#### Mode D: Gesture Recognition
- Draw rough shape, system recognizes intent
- "That looks like an ascending line" → Creates clean arc
- Tolerant of imprecise input
- BCI-friendly: Approximate input → Clean output

#### Mode E: Scanning Mode (Single-Switch)
- Highlight rows/columns sequentially
- User activates switch when desired option is highlighted
- Standard accessibility pattern
- BCI-friendly: Works with single binary signal

#### Mode F: Voice/Sound Control (Future)
- Hum pitch to draw pitch
- Duration of sound = arc length
- "Higher" / "Lower" voice commands
- BCI-adjacent: Uses vocalization, not motor control

### UI Adaptation
- "Accessibility Mode" toggle in settings
- Enlarges all UI elements
- Simplifies toolbar to essential functions
- Enables scanning navigation

### Dwell-Click Support
- Hover over button for X seconds = click
- Configurable dwell time
- Visual feedback (filling circle)
- Essential for eye-gaze users

---

## 9. Control Signal Integration

**Priority:** High  
**Status:** Planned

### Concept
Accept control signals from external sources beyond mouse.

### Supported Inputs (Planned)

#### OSC (Open Sound Control)
- Standard protocol for music/art software
- Many BCI systems can output OSC
- `/upic/cursor/x` `/upic/cursor/y` `/upic/select`

#### MIDI
- MIDI CC for continuous control
- MIDI notes for discrete actions
- Works with many adaptive controllers

#### LSL (Lab Streaming Layer)
- Standard for BCI data streaming
- Used by OpenBCI, Emotiv, g.tec, etc.
- Direct brain signal integration

#### WebSocket/HTTP API
- For custom integrations
- Web-based BCI interfaces
- Remote control

### Configuration
```yaml
# input_config.yaml
input_sources:
  - type: mouse
    enabled: true
  
  - type: osc
    enabled: true
    port: 9000
    mappings:
      cursor_x: /bci/cursor/x
      cursor_y: /bci/cursor/y
      select: /bci/select
      
  - type: lsl
    enabled: false
    stream_name: "OpenBCI_EEG"
    
  - type: midi
    enabled: false
    device: "Adaptive Controller"
```

---

## 10. Other Ideas for Future

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

