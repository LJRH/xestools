from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from PySide6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


class NormaliseDialog(QtWidgets.QDialog):
    """
    XES spectrum normalisation:
      - Toolbar (Home/Zoom/Pan)
      - Scroll-wheel zoom (x-axis) around cursor
      - Double-click to reset view
      - Draggable/resizable selection (interactive=True)
      - Span disabled while toolbar pan/zoom is active
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, parent=None, title: str = "XES Spectrum"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 560)

        self._x = np.asarray(x, dtype=float).ravel()
        self._y = np.asarray(y, dtype=float).ravel()
        ok = (np.isfinite(self._x) & np.isfinite(self._y))
        self._x = self._x[ok]
        self._y = self._y[ok]

        self._sel: Optional[Tuple[float, float]] = None
        self._area: float = float(np.trapz(self._y, self._x)) if self._x.size > 1 else 0.0
        self._hl = None  # axvspan artist for persistent highlight

        # Matplotlib UI
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        (self.line,) = self.ax.plot(self._x, self._y, lw=1.3)
        self.ax.set_xlabel("Ω (eV)")
        self.ax.set_ylabel("Intensity (counts)")
        self.ax.set_title(title)
        self._autoscale_from_data()

        # Span selector (draggable/resizable)
        self.span = self._make_span_selector(self.ax, self._on_select)

        # Info + buttons
        self.lbl = QtWidgets.QLabel(self._fmt_area())
        btn_reset = QtWidgets.QPushButton("Reset selection")
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_reset.clicked.connect(self._on_reset)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        # Layout
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.toolbar)     # toolbar for zoom/pan
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.lbl)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(btn_reset)
        row.addWidget(btn_ok)
        row.addWidget(btn_cancel)
        lay.addLayout(row)

        # Mouse/scroll bindings
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)
        self.canvas.mpl_connect("button_press_event", self._on_button_press)

        self.canvas.draw_idle()

    def _make_span_selector(self, ax, onselect):
        # Try new API (props + interactive); fallback to older (rectprops)
        try:
            span = SpanSelector(
                ax, onselect, "horizontal",
                useblit=True, interactive=True,
                props=dict(alpha=0.25, facecolor="tab:orange")
            )
            return span
        except TypeError:
            return SpanSelector(
                ax, onselect, "horizontal",
                useblit=True,
                rectprops=dict(alpha=0.25, facecolor="tab:orange")
            )

    def _on_button_press(self, event):
        # Double-click resets axes
        if event.dblclick and event.inaxes == self.ax:
            try:
                self.ax.relim(); self.ax.autoscale()
            except Exception:
                pass
            self.canvas.draw_idle()
        # Sync span active state with toolbar (avoid conflicts)
        self._sync_span_with_toolbar()

    def _autoscale_from_data(self):
        # Drop non-finite, then set limits
        x = np.asarray(self._x, dtype=float).ravel()
        y = np.asarray(self._y, dtype=float).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        if x.size:
            xmin = float(np.nanmin(x))
            xmax = float(np.nanmax(x))
            if np.isfinite([xmin, xmax]).all() and xmax > xmin:
                try:
                    self.ax.relim()
                    self.ax.autoscale_view()
                except Exception:
                    pass
                self.ax.set_xlim(xmin, xmax)
                self.ax.margins(x=0.02)
                self.canvas.draw_idle()
    
    def _sync_span_with_toolbar(self):
        # Disable span while toolbar is actively panning/zooming
        active = getattr(self.toolbar, "_active", None)
        enable_span = (active is None)
        try:
            self.span.set_active(enable_span)
        except Exception:
            pass

    def _on_scroll_zoom(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        # If toolbar is panning/zooming, let it handle the wheel
        active = getattr(self.toolbar, "_active", None)
        if active in ("PAN", "ZOOM"):
            return
        # Zoom factor
        base_scale = 1.2
        direction = +1
        st = getattr(event, "step", None)
        if st is not None:
            direction = np.sign(st) or 1
        else:
            direction = +1 if getattr(event, "button", "") == "up" else -1
        scale = (1 / base_scale) if direction > 0 else base_scale

        # Zoom around cursor (x-axis only)
        x0, x1 = self.ax.get_xlim()
        cx = event.xdata
        new_left = cx - (cx - x0) * scale
        new_right = cx + (x1 - cx) * scale
        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    def _on_select(self, xmin: float, xmax: float):
        if xmin is None or xmax is None:
            return
        a, b = sorted((xmin, xmax))
        m = (self._x >= a) & (self._x <= b)
        area = float(np.trapz(self._y[m], self._x[m])) if np.any(m) else 0.0
        self._sel = (a, b)
        self._area = area
        self.lbl.setText(self._fmt_area())
        self._update_highlight(a, b)
        self._autoscale_from_data()  # keep limits sane after redraw

    def _update_highlight(self, a: float, b: float):
        if self._hl is not None:
            try:
                self._hl.remove()
            except Exception:
                pass
            self._hl = None
        self._hl = self.ax.axvspan(a, b, color="tab:orange", alpha=0.18, lw=0)
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
        self._autoscale_from_data()
        self.canvas.draw_idle()

    def _fmt_area(self) -> str:
        if self._sel is None:
            return f"Area (full): {self._area:.6g}"
        return f"Area Ω∈[{self._sel[0]:.6g}, {self._sel[1]:.6g}] = {self._area:.6g}"

    def selected_area(self) -> float:
        return self._area

    def selected_range(self) -> Optional[Tuple[float, float]]:
        return self._sel