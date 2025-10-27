from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import QAbstractItemView
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
try:
    from lmfit.models import LinearModel, Pearson7Model
    HAVE_LMFIT = True
except Exception:
    HAVE_LMFIT = False

def _safe_interp(x_src: np.ndarray, y_src: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=float).ravel()
    y_src = np.asarray(y_src, dtype=float).ravel()
    x_new = np.asarray(x_new, dtype=float).ravel()
    if x_src.size == 0 or y_src.size == 0:
        return np.full_like(x_new, np.nan, dtype=float)
    order = np.argsort(x_src)
    xx = x_src[order]
    yy = y_src[order]
    return np.interp(x_new, xx, yy, left=np.nan, right=np.nan)

def merge_wide_and_main(
    x_main: np.ndarray, y_main: np.ndarray,
    x_wide: Optional[np.ndarray] = None, y_wide: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    xm = np.asarray(x_main, dtype=float).ravel()
    ym = np.asarray(y_main, dtype=float).ravel()
    okm = np.isfinite(xm) & np.isfinite(ym)
    xm, ym = xm[okm], ym[okm]
    if x_wide is None or y_wide is None or len(x_wide) == 0:
        return xm, ym
    xw = np.asarray(x_wide, dtype=float).ravel()
    yw = np.asarray(y_wide, dtype=float).ravel()
    okw = np.isfinite(xw) & np.isfinite(yw)
    xw, yw = xw[okw], yw[okw]
    if xm.size == 0 or xw.size == 0:
        return xm, ym
    dx = np.diff(np.unique(xm))
    step = float(np.nanmedian(dx[dx > 0])) if dx.size else max(0.1, (np.nanmax(xw) - np.nanmin(xw)) / 1000.0)
    lo_m, hi_m = float(np.nanmin(xm)), float(np.nanmax(xm))
    in_main = (xw >= lo_m) & (xw <= hi_m)
    if np.any(in_main):
        ym_on_xw = _safe_interp(xm, ym, xw[in_main])
        denom = np.where(yw[in_main] != 0.0, yw[in_main], np.nan)
        ratio = ym_on_xw / denom
        scale = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else 1.0
    else:
        scale = 1.0
    yw_scaled = yw * scale
    x_min = min(np.nanmin(xm), np.nanmin(xw))
    x_max = max(np.nanmax(xm), np.nanmax(xw))
    nsteps = max(2, int(np.round((x_max - x_min) / step)) + 1)
    x_grid = x_min + np.arange(nsteps) * step
    if x_grid[-1] < x_max - 1e-12 * max(1.0, abs(step)):
        x_grid = np.append(x_grid, x_max)
    x_union = np.union1d(np.union1d(np.round(x_grid, 8), np.round(xm, 8)), np.round(xw, 8))
    y_main_on_union = _safe_interp(xm, ym, x_union)
    y_wide_on_union = _safe_interp(xw, yw_scaled, x_union)
    y_merge = np.where(np.isfinite(y_main_on_union), y_main_on_union, y_wide_on_union)
    ok = np.isfinite(x_union) & np.isfinite(y_merge)
    return x_union[ok], y_merge[ok]

class BackgroundDialog(QtWidgets.QDialog):
    def __init__(
        self,
        x_main: np.ndarray, y_main: np.ndarray,
        x_wide: Optional[np.ndarray] = None, y_wide: Optional[np.ndarray] = None,
        parent=None, title: str = "Background Extraction"
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 720)

        # Data
        self._x, self._y = merge_wide_and_main(x_main, y_main, x_wide, y_wide)
        self._bg: Optional[np.ndarray] = None
        self._resid: Optional[np.ndarray] = None
        self._report_text: str = ""
        self._excl: List[Tuple[float, float]] = []

        # Matplotlib figure (two rows: main + residual)
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(211)
        self.axr = self.fig.add_subplot(212, sharex=self.ax)
        self._span: Optional[SpanSelector] = None
        self._cids: List[int] = []

        # ------------- Layout: slim left controls, large right plot -------------
        root = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        root.addWidget(splitter)

        # Left panel
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)

        # Controls group
        pnl = QtWidgets.QGroupBox("Fitting options")
        form = QtWidgets.QFormLayout(pnl)
        self.cb_linear = QtWidgets.QCheckBox("Linear term"); self.cb_linear.setChecked(True)
        self.cb_p7 = QtWidgets.QCheckBox("Pearson VII tail"); self.cb_p7.setChecked(False)
        self.sb_p7_center = QtWidgets.QDoubleSpinBox()
        self.sb_p7_center.setDecimals(3); self.sb_p7_center.setRange(-1e12, 1e12)
        self.sb_p7_center.setValue(float(np.nanmin(self._x) - 3.0))
        self.cb_p7_fix_center = QtWidgets.QCheckBox("Fix center")
        self.sb_p7_sigma = QtWidgets.QDoubleSpinBox()
        self.sb_p7_sigma.setDecimals(6); self.sb_p7_sigma.setRange(1e-9, 1e9); self.sb_p7_sigma.setValue(0.01)
        self.sb_p7_expon = QtWidgets.QDoubleSpinBox()
        self.sb_p7_expon.setDecimals(3); self.sb_p7_expon.setRange(0.1, 500.0); self.sb_p7_expon.setValue(2.0)
        form.addRow(self.cb_linear)
        form.addRow(self.cb_p7)
        form.addRow("P7 center (ω):", self.sb_p7_center)
        form.addRow(self.cb_p7_fix_center)
        form.addRow("P7 sigma:", self.sb_p7_sigma)
        form.addRow("P7 expon:", self.sb_p7_expon)

        # Excluded regions table + buttons
        self.tbl = QtWidgets.QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Exclude from (ω)", "to (ω)"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        try:
            self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        except AttributeError:
            self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        btn_add = QtWidgets.QPushButton("Add span (drag)")
        btn_clear = QtWidgets.QPushButton("Clear spans")
        btn_fit = QtWidgets.QPushButton("Fit background")

        btns_row = QtWidgets.QHBoxLayout()
        btns_row.addWidget(btn_add)
        btns_row.addWidget(btn_clear)

        # Report log
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("(Fit report will appear here)")
        self.report.setMinimumHeight(140)

        # OK/Cancel
        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        okrow = QtWidgets.QHBoxLayout()
        okrow.addStretch(1)
        okrow.addWidget(btn_ok)
        okrow.addWidget(btn_cancel)

        # Assemble left column
        left_layout.addWidget(pnl)
        left_layout.addWidget(QtWidgets.QLabel("Excluded regions"))
        left_layout.addWidget(self.tbl, 1)
        left_layout.addLayout(btns_row)
        left_layout.addWidget(btn_fit)
        left_layout.addWidget(QtWidgets.QLabel("Log"))
        left_layout.addWidget(self.report)
        left_layout.addStretch(1)
        left_layout.addLayout(okrow)

        # Right panel (toolbar + big canvas)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        # Add to splitter, bias width to right
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)  # left narrow
        splitter.setStretchFactor(1, 1)  # right expands

        # Initial plot and span after UI is set up
        self._plot_data()
        self._create_span()

        # Events and signals
        btn_add.clicked.connect(lambda: self._enable_span(True))
        btn_clear.clicked.connect(self._clear_spans)
        btn_fit.clicked.connect(self._fit_background)
        btn_ok.clicked.connect(self._accept_if_have_fit)
        btn_cancel.clicked.connect(self.reject)

        # Matplotlib events (store IDs for cleanup)
        self._cids.append(self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom))
        self._cids.append(self.canvas.mpl_connect("button_press_event", lambda e: self._sync_span_with_toolbar()))

    # ---------- span lifecycle ----------
    def _create_span(self):
        try:
            self._span = SpanSelector(
                self.ax, self._on_span, "horizontal",
                useblit=False, interactive=True,
                props=dict(alpha=0.25, facecolor="tab:orange")
            )
        except TypeError:
            self._span = SpanSelector(
                self.ax, self._on_span, "horizontal",
                useblit=False, rectprops=dict(alpha=0.25, facecolor="tab:orange")
            )

    def _destroy_span(self):
        if self._span is not None:
            try:
                self._span.disconnect_events()
            except Exception:
                pass
            self._span = None

    # ---------- plotting ----------
    def _autoscale_from_data(self):
        x = np.asarray(self._x, dtype=float).ravel()
        y = np.asarray(self._y, dtype=float).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        if x.size:
            xmin = float(np.nanmin(x)); xmax = float(np.nanmax(x))
            if np.isfinite([xmin, xmax]).all() and xmax > xmin:
                try:
                    self.ax.relim(); self.ax.autoscale_view()
                except Exception:
                    pass
                self.ax.set_xlim(xmin, xmax)
                self.ax.margins(x=0.02)

    def _plot_data(self):
        self._destroy_span()
        self.ax.clear()
        self.axr.clear()
        self.ax.plot(self._x, self._y, color="gray", lw=1.0, label="Merged XES")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Background fit (top) and residual (bottom)")
        self.ax.legend(loc="best", fontsize=9)
        self.axr.set_xlabel("ω (eV)")
        self.axr.set_ylabel("Residual")
        self._autoscale_from_data()
        self.canvas.draw_idle()
        self._create_span()

    # ---------- spans/table ----------
    def _enable_span(self, active: bool):
        try:
            if self._span is not None:
                self._span.set_active(active)
        except Exception:
            pass

    def _on_span(self, x0: float, x1: float):
        if x0 is None or x1 is None:
            return
        a, b = sorted((float(x0), float(x1)))
        self._excl.append((a, b))
        self._append_row(a, b)
        self._enable_span(False)

    def _append_row(self, a: float, b: float):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        for c, val in enumerate((a, b)):
            it = QtWidgets.QTableWidgetItem(f"{val:.6g}")
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, c, it)

    def _clear_spans(self):
        self._excl.clear()
        self.tbl.setRowCount(0)
        self._plot_data()

    def _read_spans(self) -> List[Tuple[float, float]]:
        spans: List[Tuple[float, float]] = []
        for r in range(self.tbl.rowCount()):
            try:
                a = float(self.tbl.item(r, 0).text())
                b = float(self.tbl.item(r, 1).text())
            except Exception:
                continue
            if np.isfinite(a) and np.isfinite(b):
                aa, bb = sorted((a, b))
                spans.append((aa, bb))
        return spans

    # ---------- toolbar/zoom ----------
    def _sync_span_with_toolbar(self):
        active = getattr(self.toolbar, "_active", None)
        enable_span = (active is None)
        self._enable_span(enable_span)

    def _on_scroll_zoom(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        active = getattr(self.toolbar, "_active", None)
        if active in ("PAN", "ZOOM"):
            return
        base_scale = 1.2
        st = getattr(event, "step", None)
        direction = (np.sign(st) or 1) if st is not None else (+1 if getattr(event, "button", "") == "up" else -1)
        scale = (1 / base_scale) if direction > 0 else base_scale
        x0, x1 = self.ax.get_xlim()
        cx = event.xdata
        new_left = cx - (cx - x0) * scale
        new_right = cx + (x1 - cx) * scale
        self.ax.set_xlim(new_left, new_right)
        self.canvas.draw_idle()

    # ---------- fitting ----------
    def _fit_background(self):
        if not HAVE_LMFIT:
            QtWidgets.QMessageBox.warning(self, "lmfit missing", "Install lmfit: pip install lmfit")
            return
        spans = self._read_spans()
        x = self._x.copy(); y = self._y.copy()
        if len(spans) == 0:
            QtWidgets.QMessageBox.information(self, "No excluded region",
                                              "Add at least one excluded region (drag spans or enter bounds).")
            return
        mask = np.ones_like(x, dtype=bool)
        for (a, b) in spans:
            mask &= ~((x >= a) & (x <= b))
        xb = x[mask]; yb = y[mask]
        ok = np.isfinite(xb) & np.isfinite(yb)
        xb, yb = xb[ok], yb[ok]
        if xb.size < 4:
            QtWidgets.QMessageBox.warning(self, "Insufficient points",
                                          "Not enough finite background points outside excluded region.")
            return
        model = None; params = None
        if self.cb_linear.isChecked():
            model = LinearModel(prefix="lin_")
            params = model.make_params(lin_slope=0.0, lin_intercept=float(np.nanmean(yb)))
        if self.cb_p7.isChecked():
            p7 = Pearson7Model(prefix="p7_")
            ppars = p7.make_params(
                p7_amplitude=float(np.nanmax(yb)),
                p7_center=float(self.sb_p7_center.value()),
                p7_sigma=float(self.sb_p7_sigma.value()),
                p7_expon=float(self.sb_p7_expon.value()),
            )
            if self.cb_p7_fix_center.isChecked():
                ppars["p7_center"].set(vary=False)
            if model is None:
                model = p7; params = ppars
            else:
                model = model + p7; params.update(ppars)
        if model is None:
            QtWidgets.QMessageBox.information(self, "No model",
                                              "Select at least one model term (Linear and/or Pearson VII).")
            return
        try:
            result = model.fit(yb, params, x=xb)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Fit error", f"lmfit failed:\n{e}")
            return

        ok_full = np.isfinite(x) & np.isfinite(y)
        xf = x[ok_full]
        y_bg_full = result.eval(result.params, x=xf)
        y_res_full = y[ok_full] - y_bg_full

        self._plot_data()
        self.ax.plot(xf, y_bg_full, "k-", lw=1.6, label="Background fit")
        for (a, b) in spans:
            self.ax.axvspan(a, b, color="tab:blue", alpha=0.15, lw=0)
        self.ax.legend(loc="best", fontsize=9)
        self.axr.plot(xf, y_res_full, color="tab:purple", lw=1.1)
        self.canvas.draw_idle()

        try:
            txt = result.fit_report()
        except Exception:
            txt = "Fit complete (report unavailable)."
        self.report.setPlainText(txt)
        self._report_text = txt
        self._x = xf; self._bg = y_bg_full; self._resid = y_res_full

    def _accept_if_have_fit(self):
        if self._bg is None or self._resid is None:
            QtWidgets.QMessageBox.information(self, "No result",
                                              "Click 'Fit background' and review the result, then press OK.")
            return
        self.accept()

    # ---------- teardown ----------
    def closeEvent(self, event):
        self._destroy_span()
        for cid in self._cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids.clear()
        super().closeEvent(event)

    # ---------- results API ----------
    def result_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self._x.copy()
        y_bg = self._bg.copy() if self._bg is not None else np.full_like(x, np.nan)
        y_res = self._resid.copy() if self._resid is not None else np.full_like(x, np.nan)
        return x, y_bg, y_res

    def get_fit_report(self) -> str:
        return self._report_text