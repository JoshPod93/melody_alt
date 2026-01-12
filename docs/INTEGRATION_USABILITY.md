# Usability/UX Integration Guide

## Overview

This guide explains how to integrate usability improvements and patient-focused features into the BCI-UPIC system. This integration may involve significant changes to the stimulus interface, visual design, interaction patterns, and workflow to make the system more accessible for patients with disabilities.

## Current Architecture

The user interface is primarily located in `src/bci/interface.py` and consists of:

1. **`BCICompositionWindow`** - Main application window
2. **`FlickerWidget`** - SSVEP flickering target display
3. **`CompositionCanvas`** - Canvas showing cursor trail and score
4. **`SSVEPStimulus`** - Stimulus timing and control (in `src/bci/stimulus.py`)

### Key Components

```
BCICompositionWindow
├── FlickerWidget (top) - 15Hz target
├── CompositionCanvas - Main drawing area
├── FlickerWidget (bottom) - 10Hz target
└── Control Panel - Buttons, status, etc.
```

## Integration Strategy

Since usability changes may be extensive, we recommend a **modular approach**:

1. **Create new UI components** alongside existing ones
2. **Add configuration options** to toggle between modes
3. **Preserve existing functionality** for backwards compatibility
4. **Gradually migrate** features to new components

## Major Integration Points

### 1. Stimulus Interface (`src/bci/stimulus.py`)

The stimulus system controls the flickering targets. Changes here may be needed for:
- Larger target sizes
- Different colors/contrasts
- Additional visual feedback
- Accessibility features (high contrast, reduced motion, etc.)

#### Current Structure

```python
@dataclass
class FlickerTarget:
    frequency: float
    phase_offset: float
    position: str
    color_on: Tuple[int, int, int]
    color_off: Tuple[int, int, int]
    size: Tuple[int, int]
```

#### Integration Approach

**Option A: Extend Existing Classes**

```python
# In src/bci/stimulus.py or new file src/bci/stimulus_accessible.py

@dataclass
class AccessibleFlickerTarget(FlickerTarget):
    """Extended flicker target with accessibility features."""
    
    # New accessibility options
    high_contrast: bool = True
    border_width: int = 5
    border_color: Tuple[int, int, int] = (255, 255, 0)  # Yellow border
    text_label: Optional[str] = None
    icon_path: Optional[str] = None
    reduced_motion: bool = False  # For motion sensitivity
    
    def get_color(self, current_time: Optional[float] = None) -> Tuple[int, int, int]:
        """Override to support high contrast mode."""
        base_color = super().get_color(current_time)
        
        if self.high_contrast:
            # Ensure maximum contrast
            if base_color == self.color_on:
                return (255, 255, 255)  # Pure white
            else:
                return (0, 0, 0)  # Pure black
        
        return base_color
```

**Option B: Configuration-Based Approach**

```python
# In src/bci/stimulus.py

@dataclass
class SSVEPStimulus:
    # ... existing fields ...
    
    # New accessibility configuration
    accessibility_config: Optional['AccessibilityConfig'] = None
    
    def __post_init__(self):
        # ... existing init ...
        
        # Apply accessibility settings
        if self.accessibility_config:
            self._apply_accessibility_settings()

@dataclass
class AccessibilityConfig:
    """Configuration for accessibility features."""
    high_contrast: bool = False
    large_targets: bool = False
    target_size_multiplier: float = 1.5
    border_visible: bool = True
    border_color: Tuple[int, int, int] = (255, 255, 0)
    text_labels: bool = True
    reduced_motion: bool = False
    audio_feedback: bool = False
```

### 2. Visual Interface (`src/bci/interface.py`)

The main interface window contains all UI elements. Changes may include:
- Layout modifications
- Additional feedback elements
- Simplified controls
- Progress indicators
- Help text/instructions

#### Integration Approach

**Create a New Accessible Window Class**

```python
# In src/bci/interface.py or new file src/bci/interface_accessible.py

class AccessibleBCICompositionWindow(BCICompositionWindow):
    """Accessible version of BCI composition window."""
    
    def __init__(self):
        # Initialize parent
        super().__init__()
        
        # Apply accessibility modifications
        self._apply_accessibility_modifications()
    
    def _apply_accessibility_modifications(self):
        """Apply accessibility features to the interface."""
        # 1. Increase target sizes
        self._enlarge_targets()
        
        # 2. Add high contrast mode
        self._enable_high_contrast()
        
        # 3. Add text labels
        self._add_text_labels()
        
        # 4. Simplify controls
        self._simplify_controls()
        
        # 5. Add progress feedback
        self._add_progress_indicators()
        
        # 6. Add help/instructions
        self._add_help_text()
    
    def _enlarge_targets(self):
        """Make flickering targets larger."""
        # Access the FlickerWidget instances
        for widget in self.findChildren(FlickerWidget):
            # Increase size
            widget.setMinimumSize(400, 150)  # Larger than default 300x80
            widget.setMaximumHeight(200)
    
    def _enable_high_contrast(self):
        """Enable high contrast mode."""
        # Modify target colors
        self.stimulus.top_target.color_on = (255, 255, 255)  # Pure white
        self.stimulus.top_target.color_off = (0, 0, 0)  # Pure black
        self.stimulus.bottom_target.color_on = (255, 255, 255)
        self.stimulus.bottom_target.color_off = (0, 0, 0)
        
        # Update canvas colors
        self.canvas.bg_color = QColor(255, 255, 255)  # White background
        self.canvas.trail_color = QColor(0, 0, 255)  # Blue trail
    
    def _add_text_labels(self):
        """Add text labels to targets."""
        # Modify FlickerWidget to show labels
        top_widget = self.findChild(FlickerWidget, "top_target")
        if top_widget:
            top_widget.set_label("LOOK UP")
        
        bottom_widget = self.findChild(FlickerWidget, "bottom_target")
        if bottom_widget:
            bottom_widget.set_label("LOOK DOWN")
    
    def _simplify_controls(self):
        """Simplify the control panel."""
        # Hide advanced options
        # Show only essential controls
        # Make buttons larger
        pass
    
    def _add_progress_indicators(self):
        """Add visual progress indicators."""
        # Add progress bar
        # Add time remaining display
        # Add completion percentage
        pass
    
    def _add_help_text(self):
        """Add help text and instructions."""
        # Add instruction label
        # Add tooltips
        # Add help button
        pass
```

**Or Use Configuration Flags**

```python
# In BCICompositionWindow.__init__()

class BCICompositionWindow(QMainWindow):
    # Add accessibility mode flag
    ACCESSIBILITY_MODE = False  # Set to True to enable
    
    def __init__(self, accessibility_mode: bool = False):
        super().__init__()
        
        self.accessibility_mode = accessibility_mode or self.ACCESSIBILITY_MODE
        
        # ... existing initialization ...
        
        if self.accessibility_mode:
            self._setup_accessibility_features()
```

### 3. FlickerWidget Modifications

The `FlickerWidget` class renders the flickering targets. You may need to modify it for:
- Larger sizes
- Borders/outlines
- Text labels
- Icons
- Different rendering styles

#### Integration Approach

**Extend FlickerWidget**

```python
# In src/bci/interface.py

class AccessibleFlickerWidget(FlickerWidget):
    """Accessible version of flicker widget."""
    
    def __init__(
        self,
        frequency: float,
        position: str = "top",
        label: Optional[str] = None,
        show_border: bool = True,
        border_color: QColor = QColor(255, 255, 0),
        parent: Optional[QWidget] = None
    ):
        super().__init__(frequency, position, parent)
        self.label_text = label
        self.show_border = show_border
        self.border_color = border_color
        
        # Larger default size
        self.setMinimumSize(400, 150)
        self.setMaximumHeight(200)
    
    def paintEvent(self, event) -> None:
        """Paint with accessibility features."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw border if enabled
        if self.show_border:
            pen = QPen(self.border_color, 5)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect())
        
        # Call parent paint (draws flickering)
        super().paintEvent(event)
        
        # Draw label text
        if self.label_text:
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            font = QFont("Arial", 24, QFont.Weight.Bold)
            painter.setFont(font)
            
            # Center text
            text_rect = self.rect()
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.label_text)
```

### 4. Workflow Modifications

Patient usability may require changes to the overall workflow:
- Simplified calibration
- Longer rest periods
- Clearer instructions
- Progress feedback
- Error recovery

#### Integration Approach

**Add Workflow Configuration**

```python
# In src/bci/interface.py

@dataclass
class PatientWorkflowConfig:
    """Configuration for patient-friendly workflow."""
    # Calibration
    calibration_trials_per_frequency: int = 5  # More trials for reliability
    calibration_rest_duration: float = 3.0  # Longer rest
    calibration_show_instructions: bool = True
    
    # Composition
    composition_duration: float = 15.0  # Longer duration
    show_progress: bool = True
    show_time_remaining: bool = True
    auto_playback: bool = True  # Auto-play after completion
    
    # Feedback
    show_confidence_feedback: bool = True
    show_direction_feedback: bool = True
    audio_feedback: bool = False
    
    # Error handling
    allow_restart: bool = True
    max_restarts: int = 3
    show_help_on_error: bool = True

class BCICompositionWindow(QMainWindow):
    def __init__(self, workflow_config: Optional[PatientWorkflowConfig] = None):
        super().__init__()
        
        self.workflow_config = workflow_config or PatientWorkflowConfig()
        
        # Apply workflow settings
        if self.workflow_config.composition_duration != 10.0:
            self.stimulus.duration = self.workflow_config.composition_duration
            self.controller.duration = self.workflow_config.composition_duration
```

### 5. Specific Usability Features

#### Feature 1: Larger Targets

**Location**: `FlickerWidget` in `src/bci/interface.py`

**Implementation**:

```python
# Option 1: Modify existing widget
def _setup_ui(self):
    # ... existing code ...
    
    # Make targets larger
    if self.accessibility_mode:
        self.top_flicker.setMinimumSize(500, 200)  # Much larger
        self.bottom_flicker.setMinimumSize(500, 200)
```

#### Feature 2: High Contrast Mode

**Location**: `FlickerWidget.paintEvent()` and `SSVEPStimulus`

**Implementation**:

```python
# In FlickerWidget
def set_high_contrast(self, enabled: bool):
    """Enable high contrast mode."""
    if enabled:
        self.color_on = QColor(255, 255, 255)  # Pure white
        self.color_off = QColor(0, 0, 0)  # Pure black
    else:
        self.color_on = QColor(255, 255, 255)  # White
        self.color_off = QColor(30, 30, 30)  # Dark gray
    self.update()
```

#### Feature 3: Text Labels

**Location**: `FlickerWidget.paintEvent()`

**Implementation**:

```python
# In FlickerWidget.paintEvent()
def paintEvent(self, event):
    # ... existing flicker drawing ...
    
    # Draw label
    if self.label_text:
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        font = QFont("Arial", 32, QFont.Weight.Bold)
        painter.setFont(font)
        text_rect = self.rect()
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.label_text)
```

#### Feature 4: Progress Indicators

**Location**: `BCICompositionWindow._setup_ui()`

**Implementation**:

```python
def _setup_ui(self):
    # ... existing UI setup ...
    
    # Add progress bar
    self.progress_bar = QProgressBar()
    self.progress_bar.setRange(0, 100)
    self.progress_bar.setValue(0)
    layout.addWidget(self.progress_bar)
    
    # Add time remaining label
    self.time_label = QLabel("Time remaining: 10s")
    layout.addWidget(self.time_label)

def _update_composition(self):
    # ... existing update code ...
    
    # Update progress
    if self.workflow_config.show_progress:
        progress = int(self.controller.progress * 100)
        self.progress_bar.setValue(progress)
        
        # Update time remaining
        if self.workflow_config.show_time_remaining:
            remaining = self.controller.duration - self.controller.position.time
            self.time_label.setText(f"Time remaining: {remaining:.1f}s")
```

#### Feature 5: Simplified Controls

**Location**: `BCICompositionWindow._setup_ui()`

**Implementation**:

```python
def _setup_controls(self):
    """Setup control panel."""
    if self.accessibility_mode:
        # Simplified controls - only essential buttons
        self.start_btn = QPushButton("START")
        self.start_btn.setMinimumSize(200, 60)  # Large button
        self.start_btn.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        
        self.calibrate_btn = QPushButton("CALIBRATE")
        self.calibrate_btn.setMinimumSize(200, 60)
        self.calibrate_btn.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        
        # Hide advanced options
        # self.advanced_group.hide()
    else:
        # Full controls
        # ... existing code ...
```

### 6. Configuration System

Create a configuration system to easily toggle features:

```python
# In src/bci/config.py (new file)

@dataclass
class AccessibilitySettings:
    """Accessibility and usability settings."""
    # Visual
    high_contrast: bool = False
    large_targets: bool = False
    target_size_multiplier: float = 1.5
    show_borders: bool = True
    border_color: Tuple[int, int, int] = (255, 255, 0)
    show_labels: bool = True
    
    # Interaction
    simplified_controls: bool = False
    auto_playback: bool = True
    show_progress: bool = True
    show_time_remaining: bool = True
    
    # Workflow
    longer_rest_periods: bool = True
    more_calibration_trials: bool = True
    allow_restart: bool = True
    
    # Feedback
    show_confidence: bool = True
    show_direction: bool = True
    audio_feedback: bool = False
    
    @classmethod
    def from_file(cls, path: str) -> 'AccessibilitySettings':
        """Load from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_file(self, path: str) -> None:
        """Save to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

# Usage in interface
def __init__(self, settings: Optional[AccessibilitySettings] = None):
    self.settings = settings or AccessibilitySettings()
    
    if self.settings.large_targets:
        self._enlarge_targets()
    
    if self.settings.high_contrast:
        self._enable_high_contrast()
    # etc.
```

### 7. Testing Your Integration

#### Test 1: Visual Changes

```python
# Test that targets are larger
window = AccessibleBCICompositionWindow()
assert window.top_flicker.minimumWidth() >= 400
assert window.top_flicker.minimumHeight() >= 150
```

#### Test 2: High Contrast

```python
# Test high contrast mode
window = AccessibleBCICompositionWindow()
window._enable_high_contrast()
assert window.stimulus.top_target.color_on == (255, 255, 255)
assert window.stimulus.top_target.color_off == (0, 0, 0)
```

#### Test 3: Workflow

```python
# Test patient workflow
config = PatientWorkflowConfig(
    composition_duration=15.0,
    show_progress=True
)
window = BCICompositionWindow(workflow_config=config)
assert window.stimulus.duration == 15.0
assert window.progress_bar is not None
```

### 8. Integration Checklist

- [ ] Identify all UI components that need modification
- [ ] Create new accessible components or extend existing ones
- [ ] Add configuration system for toggling features
- [ ] Test visual changes (sizes, colors, contrast)
- [ ] Test workflow changes (calibration, composition)
- [ ] Test with actual patients (if possible)
- [ ] Document all new features and settings
- [ ] Ensure backwards compatibility (existing code still works)
- [ ] Update main entry point to support accessibility mode
- [ ] Create example configuration files

### 9. Backwards Compatibility

To ensure existing code continues to work:

```python
# In bci_main.py or interface entry point

def run_bci_app(accessibility_mode: bool = False):
    """Run BCI application with optional accessibility mode."""
    app = QApplication(sys.argv)
    
    if accessibility_mode:
        from .interface_accessible import AccessibleBCICompositionWindow
        window = AccessibleBCICompositionWindow()
    else:
        from .interface import BCICompositionWindow
        window = BCICompositionWindow()
    
    window.show()
    sys.exit(app.exec())
```

### 10. Common Usability Improvements

Here are some common improvements you might want to implement:

1. **Larger Targets**: Increase size to 400x150 or larger
2. **High Contrast**: Pure white/black instead of gray
3. **Borders**: Yellow borders around targets for visibility
4. **Labels**: "LOOK UP" and "LOOK DOWN" text
5. **Progress Bar**: Visual progress indicator
6. **Time Remaining**: Countdown timer
7. **Simplified Controls**: Only essential buttons, larger size
8. **Instructions**: On-screen help text
9. **Audio Feedback**: Optional beeps for direction changes
10. **Error Recovery**: Clear error messages and restart options

### 11. Getting Help

If you encounter issues:

1. Review the existing `BCICompositionWindow` implementation
2. Check PyQt6 documentation for widget customization
3. Test changes incrementally (one feature at a time)
4. Use the configuration system to toggle features easily
5. Consider creating a separate "accessible" branch for major changes

---

**Important Notes**:

- **Preserve Core Functionality**: Ensure BCI control still works correctly
- **Test Thoroughly**: Visual changes shouldn't break classification
- **Document Changes**: Update user documentation for new features
- **Consider Performance**: Larger targets and additional rendering may impact performance

**Questions?** Contact the project maintainer or refer to the main project documentation.
