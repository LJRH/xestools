from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget, QLabel,
    QPushButton, QGroupBox, QFormLayout, QSizePolicy, QDoubleSpinBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


class NormaliseDialog(QDialog):
    """
    Large interactive plot on the right; slim controls on the left.
    - Drag on the right plot to select a normalisation range.
    - Or enter the range manually; plot and area update accordingly.
    API:
      selected_area() -> float
      selected_range() -> Optional[Tuple[float, float]]
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, parent=None, title: str = "Select area"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 650)

        # Data (sanitize and sort)
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]; y = y[ok]
        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]

        # Selection state (start with no selection per requirement)
        self._sel: Optional[Tuple[float, float]] = None
        self._area: Optional[float] = None
        self._suppress_spin: bool = False  # avoid feedback when updating spins programmatically

        # Precompute bounds
        self._xmin = float(np.nanmin(self.x)) if self.x.size else 0.0
        self._xmax = float(np.nanmax(self.x)) if self.x.size else 1.0

        # Layout with splitter
        root = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal, self)
        root.addWidget(splitter)

        # Right: large plot (create first so self.ax exists before any draw)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(7, 5), constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas)

        # Left: controls and small overview
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        ctrl_box = QGroupBox("Selection")
        form = QFormLayout(ctrl_box)
        self.lbl_range = QLabel("ω range: —")
        self.lbl_area = QLabel("Area: —")

        # Manual range editors (1 decimal place)
        self.spn_x1 = QDoubleSpinBox()
        self.spn_x2 = QDoubleSpinBox()
        for spn in (self.spn_x1, self.spn_x2):
            spn.setDecimals(1)
            spn.setSingleStep(0.1)
            spn.setRange(-1e12, 1e12)

        if self.x.size:
            # Set bounds (no initial selection)
            self.spn_x1.setRange(self._xmin, self._xmax)
            self.spn_x2.setRange(self._xmin, self._xmax)
            self.spn_x1.setValue(self._xmin)
            self.spn_x2.setValue(self._xmax)

        form.addRow("From ω (eV):", self.spn_x1)
        form.addRow("To ω (eV):", self.spn_x2)
        form.addRow("Range:", self.lbl_range)
        form.addRow("Area:", self.lbl_area)

        btns_row = QHBoxLayout()
        self.btn_clear = QPushButton("Clear")
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        btns_row.addWidget(self.btn_clear)
        btns_row.addStretch(1)
        btns_row.addWidget(self.btn_ok)
        btns_row.addWidget(self.btn_cancel)

        left_layout.addWidget(ctrl_box)

        # Small overview plot
        self.fig_small = Figure(figsize=(4, 2.2), constrained_layout=True)
        self.ax_small = self.fig_small.add_subplot(111)
        self.canvas_small = FigureCanvas(self.fig_small)
        self.canvas_small.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        left_layout.addWidget(self.canvas_small)

        left_layout.addLayout(btns_row)
        left_layout.addStretch(1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Initial draw with proper autoscale and no selection
        self._draw_main(autoscale=True)
        self._draw_small()
        self._update_labels()  # ensure "—" shown

        # Span selector (after first draw)
        self._span = SpanSelector(
            self.ax,
            onselect=self._on_span_selected,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.15, facecolor="tab:green"),
            interactive=True,
            drag_from_anywhere=True,
        )

        # Signals (hook after initial draw to avoid accidental triggers)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_clear.clicked.connect(self._clear_selection)
        # Recalculate area and update visuals on both change types
        self.spn_x1.valueChanged.connect(self._on_spin_changed)
        self.spn_x2.valueChanged.connect(self._on_spin_changed)
        self.spn_x1.editingFinished.connect(self._on_spin_changed)
        self.spn_x2.editingFinished.connect(self._on_spin_changed)

    # ---------- plotting ----------
    def _autoscale_axes(self):
        if self.x.size == 0:
            return
        xmin = float(np.nanmin(self.x)); xmax = float(np.nanmax(self.x))
        ymin = float(np.nanmin(self.y)); ymax = float(np.nanmax(self.y))
        if not (np.isfinite([xmin, xmax, ymin, ymax]).all()) or xmax <= xmin:
            return
        xr = xmax - xmin
        yr = ymax - ymin if ymax > ymin else (abs(ymax) + 1.0)
        self.ax.set_xlim(xmin - 0.02 * xr, xmax + 0.02 * xr)
        self.ax.set_ylim(ymin - 0.05 * yr, ymax + 0.05 * yr)

    def _draw_main(self, autoscale: bool = False):
        self.ax.clear()
        self.ax.plot(self.x, self.y, color="tab:blue", lw=1.2)
        self.ax.set_xlabel("ω (eV)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Drag to select integration range or edit values on the left")
        self.ax.grid(True, alpha=0.2)
        if autoscale:
            self._autoscale_axes()
        if self._sel:
            self.ax.axvspan(self._sel[0], self._sel[1], color="tab:green", alpha=0.15)
        self.canvas.draw_idle()

    def _draw_small(self):
        self.ax_small.clear()
        self.ax_small.plot(self.x, self.y, color="tab:gray", lw=0.9)
        self.ax_small.set_xlabel("ω (eV)")
        self.ax_small.set_ylabel("Overview")
        self.ax_small.grid(True, alpha=0.2)
        if self._sel:
            self.ax_small.axvspan(self._sel[0], self._sel[1], color="tab:green", alpha=0.15)
        self.canvas_small.draw_idle()

    # ---------- selection / area ----------
    def _compute_area(self, xlo: float, xhi: float) -> Optional[float]:
        if xhi <= xlo:
            return None
        mask = (self.x >= xlo) & (self.x <= xhi)
        if not np.any(mask):
            return None
        return float(np.trapz(self.y[mask], self.x[mask]))

    def _apply_manual_range(self, xlo: float, xhi: float, update_spins: bool = True):
        if not np.isfinite(xlo) or not np.isfinite(xhi):
            return
        # Normalise order
        if xhi < xlo:
            xlo, xhi = xhi, xlo
        self._sel = (xlo, xhi)
        self._area = self._compute_area(xlo, xhi)
        if update_spins:
            self._suppress_spin = True
            try:
                self.spn_x1.setValue(xlo)
                self.spn_x2.setValue(xhi)
            finally:
                self._suppress_spin = False
        self._update_labels()
        self._draw_main(autoscale=False)
        self._draw_small()

    def _on_span_selected(self, x1: float, x2: float):
        if not np.isfinite(x1) or not np.isfinite(x2):
            return
        xlo, xhi = (x1, x2) if x1 <= x2 else (x2, x1)
        self._apply_manual_range(xlo, xhi, update_spins=True)

    def _on_spin_changed(self):
        if self._suppress_spin:
            return
        # Always recompute area and redraw on manual input
        xlo = float(self.spn_x1.value())
        xhi = float(self.spn_x2.value())
        self._apply_manual_range(xlo, xhi, update_spins=False)

    def _clear_selection(self):
        self._sel = None
        self._area = None
        self._update_labels()
        self._draw_main(autoscale=True)
        self._draw_small()

    def _update_labels(self):
        if self._sel and self._area is not None:
            self.lbl_range.setText(f"ω range: {self._sel[0]:.1f} – {self._sel[1]:.1f} eV")
            self.lbl_area.setText(f"Area: {self._area:.6g}")
        else:
            self.lbl_range.setText("ω range: —")
            self.lbl_area.setText("Area: —")

    # ---------- API ----------
    def selected_area(self) -> float:
        return float(self._area) if self._area is not None and np.isfinite(self._area) else float("nan")

    def selected_range(self) -> Optional[Tuple[float, float]]:
        return None if self._sel is None else (float(self._sel[0]), float(self._sel[1]))