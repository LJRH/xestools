from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import logging
from PySide6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


class NormaliseDialog(QtWidgets.QDialog):
    """
    Modal dialog to view a 1D XES spectrum and select an energy range.
    Computes the numeric area under the curve in the selected range.
    Compatible with multiple Matplotlib versions.

    Zoom/pan features:
      - Matplotlib toolbar (Home/Zoom/Pan)
      - Scroll-wheel zoom around cursor (x-axis)
      - Double-click to reset viewr
      - SpanSelector disabled while toolbar pan/zoom is active
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, parent=None, title: str = "XES Spectrum"):
        super().__init__(parent)
        self._log = logging.getLogger("NormaliseDialog")
        self.setWindowTitle(title)
        self.resize(800, 500)

        self._x = np.asarray(x, dtype=float).ravel()
        self._y = np.asarray(y, dtype=float).ravel()
        ok = (np.isfinite(self._x) & np.isfinite(self._y))
        self._x = self._x[ok]
        self._y = self._y[ok]

        # Current selection and area
        self._sel: Optional[Tuple[float, float]] = None
        self._area: float = float(np.trapz(self._y, self._x)) if self._x.size > 1 else 0.0

        # Matplotlib figure
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot(self._x, self._y, lw=1.3)
        self.ax.set_xlabel("Ω (eV)")
        self.ax.set_ylabel("Intensity (counts)")
        self.ax.set_title(title)

        # Persistent highlight for selected band (independent of SpanSelector behavior)
        self._hl = None  # matplotlib artist from axvspan

        # Track mpl connection ids so we can disconnect on close
        self._mpl_cids = []

        # Span selector with compatibility across versions
        self._make_span_selector()

        # --- Toolbar & zoom/pan additions ------------------------------------
        self.toolbar = NavigationToolbar(self.canvas, self)
        # Track toolbar actions to toggle SpanSelector activity accordingly

        # Track toolbar actions to toggle SpanSelector activity accordingly
        # (toolbar._active is 'PAN'/'ZOOM'/None in NavigationToolbar2)
        self.toolbar.actionTriggered.connect(self._defer_sync_span_with_toolbar)
        # Also catch key presses that may toggle modes
        self.canvas.mpl_connect("key_press_event", self._on_key_event_toolbar_sync)

        # Scroll wheel zoom and double-click reset
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("button_press_event", self._on_button_press_reset)
        # ---------------------------------------------------------------------

        # Buttons and info
        self.lbl = QtWidgets.QLabel(self._fmt_area())
        self.btn_reset = QtWidgets.QPushButton("Reset selection")
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")

        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        # Layout
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_ok)
        btn_row.addWidget(self.btn_cancel)

        lay = QtWidgets.QVBoxLayout(self)
        # Place toolbar above the canvas
        lay.addWidget(self.toolbar)           # <--- added
        lay.addWidget(self.canvas)
        lay.addWidget(self.lbl)
        lay.addLayout(btn_row)

        # Ensure initial limits are sensible
        self._set_full_view()
        self.canvas.draw_idle()

    # ------------------------- SpanSelector setup -------------------------
    def _make_span_selector(self):
        # Try new API (props + interactive)
        try:
            self.span = SpanSelector(
                self.ax, self._on_select, "horizontal",
                useblit=True,
                interactive=True,                             # newer mpl
                props=dict(alpha=0.2, facecolor="tab:orange") # newer mpl
            )
            return
        except TypeError:
            pass
        # Fall back to older API (rectprops, no interactive)
        try:
            self.span = SpanSelector(
                self.ax, self._on_select, "horizontal",
                useblit=True,
                rectprops=dict(alpha=0.2, facecolor="tab:orange")  # older mpl
            )
        except TypeError:
            # Last resort: minimal args
            self.span = SpanSelector(self.ax, self._on_select, "horizontal")

    # ------------------------- Selection handlers -------------------------
    def _on_select(self, xmin: float, xmax: float):
        if xmin is None or xmax is None:
            return
        a, b = sorted((xmin, xmax))

        # Compute area on selected slice
        m = (self._x >= a) & (self._x <= b)
        area = float(np.trapz(self._y[m], self._x[m])) if np.any(m) else 0.0

        self._sel = (a, b)
        self._area = area
        self.lbl.setText(self._fmt_area())

        # Update persistent highlight
        if self._hl is not None:
            try:
                self._hl.remove()
            except Exception:
                pass
            self._hl = None
        self._hl = self.ax.axvspan(a, b, color="tab:orange", alpha=0.2, lw=0)
        self.canvas.draw_idle()

    def _on_reset(self):
        self._sel = None
        self._area = float(np.trapz(self._y, self._x)) if self._x.size > 1 else 0.0
        if self._hl is not None:
            try:
                self._hl.remove()
            except Exception:
                pass
            self._hl = None
        self.lbl.setText(self._fmt_area())
        self.canvas.draw_idle()

    def _fmt_area(self) -> str:
        if self._sel is None:
            return f"Area (full): {self._area:.6g}"
        return f"Area Ω∈[{self._sel[0]:.6g}, {self._sel[1]:.6g}] = {self._area:.6g}"

    def selected_area(self) -> float:
        return self._area

    def selected_range(self) -> Optional[Tuple[float, float]]:
        return self._sel

    # ------------------------- View/zoom helpers --------------------------
    def _set_full_view(self):
        """Reset axes to the full data extent and autoscale Y."""
        if self._x.size == 0:
            return
        self.ax.set_xlim(float(self._x.min()), float(self._x.max()))
        self.ax.relim()
        self.ax.autoscale_view(scaley=True)
        # keep labels/title as-is

    def _on_button_press_reset(self, event):
        """Double-click anywhere to reset the view."""
        if getattr(event, "dblclick", False):
            self._set_full_view()
            self.canvas.draw_idle()

    def _on_scroll_zoom(self, event):
        """Scroll-wheel zoom on X around the cursor."""
        # Ignore if event has no data coords (outside axes)
        if event.inaxes != self.ax:
            return
        # If toolbar is actively panning/zooming, let it handle the wheel (some backends do)
        active = getattr(self.toolbar, "_active", None)
        if active in ("PAN", "ZOOM"):
            return
        xdata = event.xdata
        if xdata is None:
            return

        # Determine scroll direction in a backend/version-robust way
        # Matplotlib >=3.6: event.step is +/-1; older: event.button in {'up','down'}
        step = getattr(event, "step", None)
        if step is None:
            direction = +1 if getattr(event, "button", "") == "up" else -1
        else:
            direction = np.sign(step) or 1

        # Zoom factor per scroll notch
        base_scale = 1.2
        scale = (1 / base_scale) if direction > 0 else base_scale

        xmin, xmax = self.ax.get_xlim()
        width = (xmax - xmin) * scale
        # Keep xdata under the cursor stationary by scaling left/right proportionally
        left = xdata - (xdata - xmin) * (width / (xmax - xmin))
        right = xdata + (xmax - xdata) * (width / (xmax - xmin))

        # Optional: clamp to full data bounds
        if self._x.size:
            xlo, xhi = float(self._x.min()), float(self._x.max())
            # avoid inverted limits
            left, right = max(min(left, right), xlo), min(max(left, right), xhi)

        self.ax.set_xlim(left, right)

        # Optionally autoscale Y to visible X range
        vis = (self._x >= left) & (self._x <= right)
        if np.any(vis):
            yslice = self._y[vis]
            if np.isfinite(yslice).any():
                ymin, ymax = np.nanmin(yslice), np.nanmax(yslice)
                if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
                    pad = 0.05 * (ymax - ymin)
                    self.ax.set_ylim(ymin - pad, ymax + pad)

        self.canvas.draw_idle()

    # -------------- Keep SpanSelector off while toolbar active ------------
    def _defer_sync_span_with_toolbar(self, *_):
        # Defer to end of event loop so toolbar._active is updated
        QtCore.QTimer.singleShot(0, self._sync_span_with_toolbar)

    def _on_key_event_toolbar_sync(self, event):
        # Toolbar toggles can also happen via keys; keep span in sync
        self._defer_sync_span_with_toolbar()

    def _sync_span_with_toolbar(self):
        """Disable SpanSelector while toolbar pan/zoom is active."""
        active = getattr(self.toolbar, "_active", None)
        enable_span = (active is None)
        try:
            self.span.set_active(enable_span)
        except Exception:
            # Older Matplotlib may not have set_active; in that case, do nothing
            pass