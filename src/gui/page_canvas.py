"""
Page Canvas for UPIC.

The main drawing area where arcs are created and edited.
This is the pitch-versus-time workspace.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
from dataclasses import dataclass

from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath,
    QMouseEvent, QWheelEvent, QKeyEvent, QFont, QFontMetrics
)

from ..core.page import Page
from ..core.arc import Arc, ArcPoint


@dataclass
class ViewTransform:
    """View transformation parameters."""
    # Visible range in data coordinates
    time_start: float = 0.0
    time_end: float = 60.0
    pitch_start: float = 0.0
    pitch_end: float = 1.0
    
    # Canvas size in pixels
    width: int = 800
    height: int = 600
    
    # Margins
    left_margin: int = 60
    right_margin: int = 20
    top_margin: int = 20
    bottom_margin: int = 40
    
    @property
    def plot_width(self) -> int:
        return self.width - self.left_margin - self.right_margin
    
    @property
    def plot_height(self) -> int:
        return self.height - self.top_margin - self.bottom_margin
    
    @property
    def time_range(self) -> float:
        return self.time_end - self.time_start
    
    @property
    def pitch_range(self) -> float:
        return self.pitch_end - self.pitch_start
    
    def time_to_x(self, time: float) -> float:
        """Convert time to x pixel coordinate."""
        if self.time_range == 0:
            return self.left_margin
        return self.left_margin + (time - self.time_start) / self.time_range * self.plot_width
    
    def x_to_time(self, x: float) -> float:
        """Convert x pixel coordinate to time."""
        if self.plot_width == 0:
            return self.time_start
        return self.time_start + (x - self.left_margin) / self.plot_width * self.time_range
    
    def pitch_to_y(self, pitch: float) -> float:
        """Convert pitch to y pixel coordinate (inverted - high pitch at top)."""
        if self.pitch_range == 0:
            return self.top_margin
        return self.top_margin + (1 - (pitch - self.pitch_start) / self.pitch_range) * self.plot_height
    
    def y_to_pitch(self, y: float) -> float:
        """Convert y pixel coordinate to pitch."""
        if self.plot_height == 0:
            return self.pitch_start
        return self.pitch_start + (1 - (y - self.top_margin) / self.plot_height) * self.pitch_range
    
    def point_to_data(self, point: QPointF) -> Tuple[float, float]:
        """Convert pixel point to (time, pitch)."""
        return (self.x_to_time(point.x()), self.y_to_pitch(point.y()))
    
    def data_to_point(self, time: float, pitch: float) -> QPointF:
        """Convert (time, pitch) to pixel point."""
        return QPointF(self.time_to_x(time), self.pitch_to_y(pitch))


class PageCanvas(QWidget):
    """
    The main drawing canvas for UPIC.
    
    Displays arcs in a pitch-versus-time coordinate system and
    handles all drawing and editing interactions.
    """
    
    # Signals
    arc_created = pyqtSignal(str)  # Arc ID
    arc_modified = pyqtSignal(str)  # Arc ID
    arc_deleted = pyqtSignal(str)  # Arc ID
    selection_changed = pyqtSignal(list)  # List of selected arc IDs
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # Data
        self.page: Optional[Page] = None
        self.transform = ViewTransform()
        
        # Tool state
        self.current_tool = "draw"  # Start in draw mode
        self.current_waveform = "Sine"
        self.current_envelope = "ADSR"
        
        # Selection
        self.selected_arc_ids: List[str] = []
        self.hovered_arc_id: Optional[str] = None
        
        # Drawing state
        self.is_drawing = False
        self.drawing_points: List[Tuple[float, float]] = []
        self.is_panning = False
        self.pan_start: Optional[QPointF] = None
        self.pan_start_transform: Optional[Tuple[float, float]] = None
        
        # Selection rectangle
        self.selection_rect: Optional[QRectF] = None
        self.selection_start: Optional[QPointF] = None
        
        # Playhead
        self.playhead_time: float = 0.0
        
        # Display settings
        self.show_grid = True
        self.arc_line_width = 2.0
        self.selected_line_width = 3.0
        
        # Colors
        self.bg_color = QColor(30, 30, 35)
        self.grid_color = QColor(60, 60, 70)
        self.axis_color = QColor(100, 100, 110)
        self.playhead_color = QColor(255, 100, 100)
        self.selection_rect_color = QColor(100, 150, 255, 50)
        self.selection_border_color = QColor(100, 150, 255)
        
        # Setup
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 300)
        
        # Cursor
        self._update_cursor()
    
    def set_page(self, page: Page) -> None:
        """Set the page to display."""
        self.page = page
        self.selected_arc_ids.clear()
        self.transform.time_end = page.settings.duration
        self.update()
    
    def set_tool(self, tool: str) -> None:
        """Set the current tool."""
        self.current_tool = tool
        self._update_cursor()
        self.update()
    
    def set_current_waveform(self, name: str) -> None:
        """Set the waveform to use for new arcs."""
        self.current_waveform = name
    
    def set_current_envelope(self, name: str) -> None:
        """Set the envelope to use for new arcs."""
        self.current_envelope = name
    
    def set_show_grid(self, show: bool) -> None:
        """Toggle grid display."""
        self.show_grid = show
        self.update()
    
    def set_playhead_position(self, time: float) -> None:
        """Set the playhead position."""
        self.playhead_time = time
        self.update()
    
    def zoom(self, factor: float, center: Optional[QPointF] = None) -> None:
        """Zoom the view by a factor."""
        if center is None:
            center = QPointF(self.width() / 2, self.height() / 2)
        
        # Get center in data coordinates
        center_time, center_pitch = self.transform.point_to_data(center)
        
        # Scale ranges
        time_range = self.transform.time_range / factor
        pitch_range = self.transform.pitch_range / factor
        
        # Recenter
        self.transform.time_start = center_time - time_range / 2
        self.transform.time_end = center_time + time_range / 2
        self.transform.pitch_start = max(0, center_pitch - pitch_range / 2)
        self.transform.pitch_end = min(1, center_pitch + pitch_range / 2)
        
        self.update()
    
    def fit_to_window(self) -> None:
        """Reset view to show entire page."""
        if self.page:
            self.transform.time_start = 0
            self.transform.time_end = self.page.settings.duration
            self.transform.pitch_start = 0
            self.transform.pitch_end = 1
            self.update()
    
    def delete_selected(self) -> None:
        """Delete selected arcs."""
        if self.page:
            self.page.save_state()
            for arc_id in self.selected_arc_ids:
                self.page.remove_arc(arc_id)
                self.arc_deleted.emit(arc_id)
            self.selected_arc_ids.clear()
            self.selection_changed.emit([])
            self.update()
    
    def _update_cursor(self) -> None:
        """Update cursor based on current tool."""
        if self.current_tool == "draw":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self.current_tool == "line":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self.current_tool == "erase":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self.current_tool == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.current_tool == "zoom":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def _get_arc_at_point(self, point: QPointF) -> Optional[str]:
        """Get the arc ID at a pixel point, if any."""
        if not self.page:
            return None
        
        time, pitch = self.transform.point_to_data(point)
        threshold = 0.02  # Pitch threshold for hit testing
        
        for arc in self.page.arcs.values():
            if arc.start_time <= time <= arc.end_time:
                arc_pitch = arc.get_pitch_at_time(time)
                if abs(arc_pitch - pitch) < threshold:
                    return arc.id
        
        return None
    
    def _get_arcs_in_rect(self, rect: QRectF) -> List[str]:
        """Get all arc IDs that intersect a rectangle."""
        if not self.page:
            return []
        
        t1, p1 = self.transform.point_to_data(rect.topLeft())
        t2, p2 = self.transform.point_to_data(rect.bottomRight())
        
        time_start = min(t1, t2)
        time_end = max(t1, t2)
        pitch_start = min(p1, p2)
        pitch_end = max(p1, p2)
        
        result = []
        for arc in self.page.arcs.values():
            if arc.overlaps_time_range(time_start, time_end):
                if arc.min_pitch <= pitch_end and arc.max_pitch >= pitch_start:
                    result.append(arc.id)
        
        return result
    
    # ==================== Event Handlers ====================
    
    def resizeEvent(self, event) -> None:
        """Handle resize."""
        self.transform.width = self.width()
        self.transform.height = self.height()
        super().resizeEvent(event)
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press."""
        pos = event.position()
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == "select":
                # Check if clicking on an arc
                arc_id = self._get_arc_at_point(pos)
                
                if arc_id:
                    # Modify selection
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                        if arc_id in self.selected_arc_ids:
                            self.selected_arc_ids.remove(arc_id)
                        else:
                            self.selected_arc_ids.append(arc_id)
                    else:
                        self.selected_arc_ids = [arc_id]
                    self.selection_changed.emit(self.selected_arc_ids)
                else:
                    # Start selection rectangle
                    if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                        self.selected_arc_ids.clear()
                        self.selection_changed.emit([])
                    self.selection_start = pos
                    self.selection_rect = QRectF(pos, pos)
            
            elif self.current_tool in ("draw", "line"):
                # Start drawing
                if self.page is None:
                    print("Warning: No page set, cannot draw!")
                    return
                self.is_drawing = True
                time, pitch = self.transform.point_to_data(pos)
                pitch = max(0, min(1, pitch))  # Clamp pitch
                self.drawing_points = [(time, pitch)]
                print(f"Started drawing at time={time:.2f}, pitch={pitch:.2f}")
            
            elif self.current_tool == "erase":
                # Erase arc under cursor
                arc_id = self._get_arc_at_point(pos)
                if arc_id and self.page:
                    self.page.save_state()
                    self.page.remove_arc(arc_id)
                    if arc_id in self.selected_arc_ids:
                        self.selected_arc_ids.remove(arc_id)
                    self.arc_deleted.emit(arc_id)
            
            elif self.current_tool == "pan":
                self.is_panning = True
                self.pan_start = pos
                self.pan_start_transform = (
                    self.transform.time_start,
                    self.transform.pitch_start
                )
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
            elif self.current_tool == "zoom":
                # Zoom in on click
                self.zoom(1.5, pos)
        
        elif event.button() == Qt.MouseButton.RightButton:
            if self.current_tool == "zoom":
                # Zoom out on right click
                self.zoom(0.67, pos)
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move."""
        pos = event.position()
        
        if self.is_drawing:
            time, pitch = self.transform.point_to_data(pos)
            pitch = max(0, min(1, pitch))  # Clamp pitch
            
            if self.current_tool == "draw":
                self.drawing_points.append((time, pitch))
            elif self.current_tool == "line":
                # Only keep first and current point for line
                if len(self.drawing_points) > 1:
                    self.drawing_points = [self.drawing_points[0], (time, pitch)]
                else:
                    self.drawing_points.append((time, pitch))
        
        elif self.selection_rect is not None:
            self.selection_rect = QRectF(self.selection_start, pos).normalized()
        
        elif self.is_panning and self.pan_start and self.pan_start_transform:
            # Calculate pan delta
            dx = pos.x() - self.pan_start.x()
            dy = pos.y() - self.pan_start.y()
            
            # Convert to data coordinates
            dt = -dx / self.transform.plot_width * self.transform.time_range
            dp = dy / self.transform.plot_height * self.transform.pitch_range
            
            self.transform.time_start = self.pan_start_transform[0] + dt
            self.transform.time_end = self.transform.time_start + self.transform.time_range
            
            new_pitch_start = self.pan_start_transform[1] + dp
            if 0 <= new_pitch_start <= 1 - self.transform.pitch_range:
                self.transform.pitch_start = new_pitch_start
                self.transform.pitch_end = new_pitch_start + self.transform.pitch_range
        
        else:
            # Update hover state
            self.hovered_arc_id = self._get_arc_at_point(pos)
        
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.is_drawing and self.page and len(self.drawing_points) >= 2:
                # Sort points by time to ensure proper order
                self.drawing_points.sort(key=lambda p: p[0])
                
                # Create new arc
                self.page.save_state()
                arc = Arc(
                    name=f"Arc {len(self.page.arcs) + 1}",
                    waveform_name=self.current_waveform,
                    envelope_name=self.current_envelope
                )
                arc.set_points_from_tuples(self.drawing_points)
                
                # Debug: print arc info
                print(f"Creating arc: {arc.name}, points: {len(arc.points)}, "
                      f"time: {arc.start_time:.2f}-{arc.end_time:.2f}, "
                      f"pitch: {arc.min_pitch:.2f}-{arc.max_pitch:.2f}")
                
                if self.page.add_arc(arc):
                    print(f"Arc added successfully. Total arcs: {len(self.page.arcs)}")
                    self.arc_created.emit(arc.id)
                    
                    # Select the new arc
                    self.selected_arc_ids = [arc.id]
                    self.selection_changed.emit(self.selected_arc_ids)
                else:
                    print("Failed to add arc!")
            
            if self.selection_rect is not None:
                # Select arcs in rectangle
                arc_ids = self._get_arcs_in_rect(self.selection_rect)
                if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                    for arc_id in arc_ids:
                        if arc_id not in self.selected_arc_ids:
                            self.selected_arc_ids.append(arc_id)
                else:
                    self.selected_arc_ids = arc_ids
                self.selection_changed.emit(self.selected_arc_ids)
                self.selection_rect = None
            
            self.is_drawing = False
            self.drawing_points.clear()
            
            if self.is_panning:
                self.is_panning = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
        
        self.update()
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.zoom(factor, event.position())
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press."""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif event.key() == Qt.Key.Key_Escape:
            self.selected_arc_ids.clear()
            self.selection_changed.emit([])
            self.update()
        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Select all
            if self.page:
                self.selected_arc_ids = list(self.page.arcs.keys())
                self.selection_changed.emit(self.selected_arc_ids)
                self.update()
        else:
            super().keyPressEvent(event)
    
    # ==================== Painting ====================
    
    def paintEvent(self, event) -> None:
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Grid
        if self.show_grid:
            self._draw_grid(painter)
        
        # Axes
        self._draw_axes(painter)
        
        # Arcs
        if self.page:
            for arc in self.page.arcs.values():
                self._draw_arc(painter, arc)
        
        # Drawing preview
        if self.is_drawing and self.drawing_points:
            self._draw_preview(painter)
        
        # Selection rectangle
        if self.selection_rect:
            painter.setPen(QPen(self.selection_border_color, 1, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(self.selection_rect_color))
            painter.drawRect(self.selection_rect)
        
        # Playhead
        self._draw_playhead(painter)
        
        painter.end()
    
    def _draw_grid(self, painter: QPainter) -> None:
        """Draw the grid."""
        painter.setPen(QPen(self.grid_color, 1))
        
        # Vertical lines (time)
        time_step = self._calculate_grid_step(self.transform.time_range, 10)
        time = (self.transform.time_start // time_step + 1) * time_step
        while time < self.transform.time_end:
            x = self.transform.time_to_x(time)
            painter.drawLine(int(x), self.transform.top_margin,
                           int(x), self.height() - self.transform.bottom_margin)
            time += time_step
        
        # Horizontal lines (pitch)
        pitch_step = self._calculate_grid_step(self.transform.pitch_range, 8)
        pitch = (self.transform.pitch_start // pitch_step + 1) * pitch_step
        while pitch < self.transform.pitch_end:
            y = self.transform.pitch_to_y(pitch)
            painter.drawLine(self.transform.left_margin, int(y),
                           self.width() - self.transform.right_margin, int(y))
            pitch += pitch_step
    
    def _draw_axes(self, painter: QPainter) -> None:
        """Draw axes and labels."""
        painter.setPen(QPen(self.axis_color, 1))
        font = QFont("Arial", 9)
        painter.setFont(font)
        
        # Left axis (pitch)
        painter.drawLine(
            self.transform.left_margin, self.transform.top_margin,
            self.transform.left_margin, self.height() - self.transform.bottom_margin
        )
        
        # Bottom axis (time)
        painter.drawLine(
            self.transform.left_margin, self.height() - self.transform.bottom_margin,
            self.width() - self.transform.right_margin, self.height() - self.transform.bottom_margin
        )
        
        # Time labels
        time_step = self._calculate_grid_step(self.transform.time_range, 10)
        time = (self.transform.time_start // time_step + 1) * time_step
        while time < self.transform.time_end:
            x = self.transform.time_to_x(time)
            label = f"{time:.1f}s"
            painter.drawText(int(x) - 15, self.height() - 10, label)
            time += time_step
        
        # Pitch labels (simplified - just show 0-1 range)
        for pitch in [0.0, 0.25, 0.5, 0.75, 1.0]:
            if self.transform.pitch_start <= pitch <= self.transform.pitch_end:
                y = self.transform.pitch_to_y(pitch)
                painter.drawText(5, int(y) + 4, f"{pitch:.2f}")
    
    def _draw_arc(self, painter: QPainter, arc: Arc) -> None:
        """Draw a single arc."""
        if not arc.points or len(arc.points) < 2:
            return
        
        # Determine color and width
        is_selected = arc.id in self.selected_arc_ids
        is_hovered = arc.id == self.hovered_arc_id
        
        color = QColor(*arc.color)
        if arc.muted:
            color.setAlpha(100)
        if is_hovered and not is_selected:
            color = color.lighter(120)
        
        width = self.selected_line_width if is_selected else self.arc_line_width
        
        # Draw path
        path = QPainterPath()
        first_point = arc.points[0]
        start = self.transform.data_to_point(first_point.time, first_point.pitch)
        path.moveTo(start)
        
        for point in arc.points[1:]:
            pos = self.transform.data_to_point(point.time, point.pitch)
            path.lineTo(pos)
        
        painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, 
                           Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(path)
        
        # Draw selection handles if selected
        if is_selected:
            painter.setBrush(QBrush(Qt.GlobalColor.white))
            painter.setPen(QPen(color, 1))
            for point in arc.points:
                pos = self.transform.data_to_point(point.time, point.pitch)
                painter.drawEllipse(pos, 4, 4)
    
    def _draw_preview(self, painter: QPainter) -> None:
        """Draw the arc being drawn."""
        if len(self.drawing_points) < 2:
            return
        
        path = QPainterPath()
        first = self.drawing_points[0]
        start = self.transform.data_to_point(first[0], first[1])
        path.moveTo(start)
        
        for time, pitch in self.drawing_points[1:]:
            pos = self.transform.data_to_point(time, pitch)
            path.lineTo(pos)
        
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2, Qt.PenStyle.DashLine))
        painter.drawPath(path)
    
    def _draw_playhead(self, painter: QPainter) -> None:
        """Draw the playhead."""
        if self.transform.time_start <= self.playhead_time <= self.transform.time_end:
            x = self.transform.time_to_x(self.playhead_time)
            painter.setPen(QPen(self.playhead_color, 2))
            painter.drawLine(
                int(x), self.transform.top_margin,
                int(x), self.height() - self.transform.bottom_margin
            )
    
    def _calculate_grid_step(self, range_val: float, target_lines: int) -> float:
        """Calculate appropriate grid step size."""
        if range_val <= 0:
            return 1.0
        
        rough_step = range_val / target_lines
        magnitude = 10 ** int(np.log10(rough_step) if rough_step > 0 else 0)
        
        normalized = rough_step / magnitude
        if normalized < 1.5:
            return magnitude
        elif normalized < 3:
            return 2 * magnitude
        elif normalized < 7:
            return 5 * magnitude
        else:
            return 10 * magnitude


# Import numpy for grid calculation
import numpy as np

