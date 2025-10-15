from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from PySide6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


class NormaliseDialog(QtWidgets.QDialog):
    """
    Modal dialog to view a 1D XES spectrum and select an energy range.
    Computes the numeric area under the curve in the selected range.
    Compatible with multiple Matplotlib versions.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, parent=None, title: str = "XES Spectrum"):
        super().__init__(parent)
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

        # Span selector with compatibility across versions
        self._make_span_selector()

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
        lay.addWidget(self.canvas)
        lay.addWidget(self.lbl)
        lay.addLayout(btn_row)

        self.canvas.draw_idle()

    def _make_span_selector(self):
        # Try new API (props + interactive)
        try:
            self.span = SpanSelector(
                self.ax, self._on_select, "horizontal",
                useblit=True,
                interactive=True,                             # newer mpl
                props=dict(alpha=0.2, facecolor="tab:orange") # newer mpl (replaces rectprops)
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