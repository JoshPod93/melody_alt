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

## 4. Other Ideas for Future

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

