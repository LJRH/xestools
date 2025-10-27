import os
import numpy as np
from typing import List, Tuple
from PySide6 import QtCore
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from i20_xes.modules.dataset import DataSet

_LINE_COLORS = ["#ffffff", "#ffcc00", "#00ccff"]
_BAND_ALPHA = 0.15

class PlotWidget(QWidget):
    lines_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        gs = self.figure.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1])
        self.ax_img = self.figure.add_subplot(gs[0, 0])
        self.ax_prof = self.figure.add_subplot(gs[1, 0])

        self._cbar = None
        self._lines: List[Line2D] = []
        self._bands: List = []  # keep handles to band artists (axvspan/axhspan)
        self._drag_idx = None
        self._line_orientation = "vertical"
        self._bandwidth = 2.0
        self._suppress_emit = False
        self._dataset: DataSet | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    # ------------ helpers to suppress signals ------------
    def set_signal_suppressed(self, state: bool):
        self._suppress_emit = bool(state)

    def _emit_lines_changed(self):
        if not self._suppress_emit:
            self.lines_changed.emit()

    # ------------ core clear/plot ------------
    def clear(self):
        # Remove colorbar first
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        # Remove bands BEFORE clearing axes to avoid residual artists
        self._remove_bands()

        # Clear axes
        self.ax_img.cla()
        self.ax_prof.cla()

        # Reset
        self._lines = []
        self._drag_idx = None

    def plot(self, dataset: DataSet | None):
        self._dataset = dataset
        self.clear()

        if dataset is None:
            self.ax_img.set_title("No data loaded")
            self.ax_prof.set_visible(True)  # keep ROI subplot visible for future RXES
            self.canvas.draw_idle()
            return

        title = os.path.basename(dataset.source) if dataset.source else "Data"

        if dataset.kind == "1D":
            # Hide profile axis for simple 1D
            self.ax_prof.set_visible(False)
            x = dataset.x if dataset.x is not None else np.arange(len(dataset.y))
            y = dataset.y
            self.ax_img.plot(x, y, lw=1.5)
            self.ax_img.set_xlabel(dataset.xlabel or "Energy")
            self.ax_img.set_ylabel(dataset.ylabel or "Intensity")
            self.ax_img.set_title(title)
        elif dataset.kind == "2D":
            # Show profile axis for RXES
            self.ax_prof.set_visible(True)

            Z = dataset.z
            if dataset.x2d is not None and dataset.y2d is not None:
                pc = self.ax_img.pcolormesh(dataset.x2d, dataset.y2d, Z, shading="auto", cmap="viridis")
                self._cbar = self.figure.colorbar(pc, ax=self.ax_img)
            else:
                extent = None
                if dataset.x is not None and dataset.y is not None:
                    try:
                        if np.all(np.diff(dataset.x) > 0) and np.all(np.diff(dataset.y) > 0):
                            extent = [float(dataset.x.min()), float(dataset.x.max()),
                                      float(dataset.y.min()), float(dataset.y.max())]
                    except Exception:
                        extent = None
                im = self.ax_img.imshow(Z, aspect='auto', origin='lower', extent=extent, cmap='viridis')
                self._cbar = self.figure.colorbar(im, ax=self.ax_img)

            if self._cbar is not None:
                self._cbar.set_label(dataset.zlabel or "Intensity")
            self.ax_img.set_xlabel(dataset.xlabel or "X")
            self.ax_img.set_ylabel(dataset.ylabel or "Y")
            self.ax_img.set_title(title)

            # Prepare profile axis
            self.ax_prof.set_ylabel("Integrated counts")
            self.ax_prof.grid(True, alpha=0.3)

            # Ensure at least one ROI line for RXES, and draw initial bands
            if len(self._lines) == 0:
                self._create_initial_line()
            self._update_bands()

        self.canvas.draw_idle()

    # ------------ XES overlays (hide ROI subplot) ------------
    def plot_xes_bundle(self, curves, avg=None, title="XES bundle"):
        self.clear()
        self.ax_prof.set_visible(False)
        self.ax_prof.cla()

        for c in curves:
            x = c.get("x"); y = c.get("y")
            if x is None or y is None:
                continue
            self.ax_img.plot(x, y, lw=1.0, alpha=c.get("alpha", 0.7),
                             color=c.get("color", None), label=c.get("label", ""))

        if avg is not None and avg.get("x") is not None:
            self.ax_img.plot(avg["x"], avg["y"], lw=2.2, color="black", label=avg.get("label", "Average"))

        self.ax_img.set_xlabel("ω (eV)")
        self.ax_img.set_ylabel("Intensity (XES)")
        self.ax_img.set_title(title)
        try:
            if len(curves) <= 12:
                self.ax_img.legend(loc="best", fontsize=9)
        except Exception:
            pass

        self.canvas.draw_idle()

    # ------------ ROI line/band API ------------
    def set_line_orientation(self, orientation: str):
        assert orientation in ("vertical", "horizontal")
        # If orientation unchanged, just refresh bands and notify
        if orientation == self._line_orientation:
            if len(self._lines) == 0:
                self._create_initial_line()
            self._update_bands()
            self._emit_lines_changed()
            return

        # Orientation changed: rebuild lines in the new orientation
        old_n = max(1, len(self._lines))
        # remove all existing lines
        for ln in self._lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._lines = []

        self._line_orientation = orientation
        # create same number of lines in the new orientation at sensible positions
        self._create_initial_line()
        while len(self._lines) < old_n:
            self._add_line_internal()

        self._update_bands()
        self.canvas.draw_idle()
        self._emit_lines_changed()

    def set_bandwidth(self, width: float):
        self._bandwidth = max(0.0, float(width))
        self._update_bands()
        self._emit_lines_changed()

    def ensure_line_count(self, n: int):
        n = max(0, min(3, n))
        if len(self._lines) == 0 and n > 0:
            self._create_initial_line()
        while len(self._lines) < n:
            self._add_line_internal()
        while len(self._lines) > n:
            ln = self._lines.pop()
            try:
                ln.remove()
            except Exception:
                pass
        self._update_bands()
        self.canvas.draw_idle()
        self._emit_lines_changed()

    def add_line(self):
        if len(self._lines) == 0:
            self._create_initial_line()
        elif len(self._lines) < 3:
            self._add_line_internal()
        self._update_bands()
        self.canvas.draw_idle()
        self._emit_lines_changed()

    def remove_line(self):
        if len(self._lines) == 0:
            return
        ln = self._lines.pop()
        try:
            ln.remove()
        except Exception:
            pass
        self._update_bands()
        self.canvas.draw_idle()
        self._emit_lines_changed()

    def get_line_positions(self) -> List[float]:
        pos = []
        for ln in self._lines:
            if self._line_orientation == "vertical":
                xdata = ln.get_xdata(orig=False)
                if xdata[0] == xdata[1]:
                    pos.append(xdata[0])
            else:
                ydata = ln.get_ydata(orig=False)
                if ydata[0] == ydata[1]:
                    pos.append(ydata[0])
        return pos

    def set_line_positions(self, positions: List[float]):
        self.ensure_line_count(len(positions))
        for ln, val in zip(self._lines, positions):
            if self._line_orientation == "vertical":
                ln.set_xdata([val, val])
            else:
                ln.set_ydata([val, val])
        self._update_bands()
        self.canvas.draw_idle()
        self._emit_lines_changed()

    def _create_initial_line(self):
        if self._line_orientation == "vertical":
            xmin, xmax = self.ax_img.get_xlim()
            x = xmin + 0.5 * (xmax - xmin)
            ln = self.ax_img.axvline(x, color=_LINE_COLORS[0], lw=1.8, alpha=0.95)
        else:
            ymin, ymax = self.ax_img.get_ylim()
            y = ymin + 0.5 * (ymax - ymin)
            ln = self.ax_img.axhline(y, color=_LINE_COLORS[0], lw=1.8, alpha=0.95)
        self._lines = [ln]

    def _add_line_internal(self):
        idx = len(self._lines)
        color = _LINE_COLORS[min(idx, len(_LINE_COLORS) - 1)]
        if self._line_orientation == "vertical":
            xmin, xmax = self.ax_img.get_xlim()
            x = xmin + (0.33 if idx == 1 else 0.66) * (xmax - xmin)
            ln = self.ax_img.axvline(x, color=color, lw=1.6, alpha=0.9, ls="--")
        else:
            ymin, ymax = self.ax_img.get_ylim()
            y = ymin + (0.33 if idx == 1 else 0.66) * (ymax - ymin)
            ln = self.ax_img.axhline(y, color=color, lw=1.6, alpha=0.9, ls="--")
        self._lines.append(ln)

    # ------------ band management ------------
    def _remove_bands(self):
        # Safely remove any existing band artists
        if not self._bands:
            return
        for b in list(self._bands):
            try:
                b.remove()
            except Exception:
                pass
        self._bands.clear()

    def _update_bands(self):
        self._remove_bands()
        if len(self._lines) == 0 or self._bandwidth <= 0:
            self.canvas.draw_idle()
            return
        half = self._bandwidth / 2.0
        for i, ln in enumerate(self._lines):
            color = _LINE_COLORS[min(i, len(_LINE_COLORS) - 1)]
            if self._line_orientation == "vertical":
                xdata = ln.get_xdata(orig=False)
                if xdata[0] != xdata[1]:
                    continue  # not a vertical line
                x0 = xdata[0]
                band = self.ax_img.axvspan(x0 - half, x0 + half, color=color, alpha=_BAND_ALPHA, lw=0)
            else:
                ydata = ln.get_ydata(orig=False)
                if ydata[0] != ydata[1]:
                    continue  # not a horizontal line
                y0 = ydata[0]
                band = self.ax_img.axhspan(y0 - half, y0 + half, color=color, alpha=_BAND_ALPHA, lw=0)
            self._bands.append(band)
        self.canvas.draw_idle()

    # ------------ mouse interaction ------------
    # _on_press, _on_motion, _on_release remain the same but call self._update_bands() after moving lines.

    def _on_press(self, event):
        if event.inaxes != self.ax_img or len(self._lines) == 0:
            return
        tol = 10.0
        for idx, ln in enumerate(self._lines):
            if self._line_orientation == "vertical":
                x0 = ln.get_xdata()[0]
                dpx = abs(self.ax_img.transData.transform((x0, 0))[0] - event.x)
                if dpx <= tol:
                    self._drag_idx = idx
                    break
            else:
                y0 = ln.get_ydata()[0]
                dpy = abs(self.ax_img.transData.transform((0, y0))[1] - event.y)
                if dpy <= tol:
                    self._drag_idx = idx
                    break

    def _on_motion(self, event):
        if self._drag_idx is None or event.inaxes != self.ax_img:
            return
        if self._line_orientation == "vertical":
            if event.xdata is None:
                return
            x = event.xdata
            self._lines[self._drag_idx].set_xdata([x, x])
        else:
            if event.ydata is None:
                return
            y = event.ydata
            self._lines[self._drag_idx].set_ydata([y, y])
        self._update_bands()
        self.canvas.draw_idle()
        self._emit_lines_changed()

    def _on_release(self, event):
        self._drag_idx = None
        self._emit_lines_changed()
        self._update_bands()

    # Profile panel

    def plot_profiles(self, x_label: str, curves: List[Tuple[np.ndarray, np.ndarray, str]]):
        self.ax_prof.clear()
        for i, (xv, yv, lbl) in enumerate(curves):
            self.ax_prof.plot(xv, yv, label=lbl)
        self.ax_prof.set_xlabel(x_label)
        self.ax_prof.set_ylabel("Integrated counts")
        if curves:
            self.ax_prof.legend(loc="best", fontsize=9)
        self.ax_prof.grid(True, alpha=0.3)
        self.canvas.draw_idle()