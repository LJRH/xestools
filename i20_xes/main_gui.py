import os
from typing import Optional, List, Tuple

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget,
    QFileDialog, QMessageBox, QInputDialog, QDialog
)

from i20_xes.modules.dataset import DataSet
from i20_xes.widgets.io_panel import IOPanel, RXESPanel
from i20_xes.widgets.xes_panel import XESPanel
from i20_xes.widgets.plot_widget import PlotWidget
from i20_xes.widgets.normalise_dialog import NormaliseDialog
from i20_xes.modules import io as io_mod
from i20_xes.modules.scan import Scan
from i20_xes.modules import i20_loader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("I20 XES/RXES Viewer")
        self.resize(1300, 900)

        # Session containers
        self.scan = Scan()
        self.current_scan_number: Optional[int] = None
        self.dataset: Optional[DataSet] = None
        self._last_profiles: List[Tuple[np.ndarray, np.ndarray, str]] = []

        # XES multi-scan workflow
        self._xes_items: List[dict] = []  # each: dict(path, channel, x, y, label)
        self._xes_avg: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._xes_avg_norm_factor: Optional[float] = None

        # Layout
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # Left: I/O + tabs (RXES | XES)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        self.io_panel = IOPanel()
        self.rxes_panel = RXESPanel()
        self.xes_panel = XESPanel()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.rxes_panel, "RXES")
        self.tabs.addTab(self.xes_panel, "XES")

        left_layout.addWidget(self.io_panel)
        left_layout.addWidget(self.tabs)
        left_layout.addStretch(1)

        # Right: Plot
        self.plot = PlotWidget()
        try:
            self.plot.lines_changed.connect(self.update_profiles)
        except Exception:
            pass

        root_layout.addWidget(left_container, 0)
        root_layout.addWidget(self.plot, 1)

        # I/O actions
        self.io_panel.btn_load.clicked.connect(self.on_load)
        self.io_panel.btn_save_ascii.clicked.connect(self.on_save_ascii)
        self.io_panel.btn_save_nexus.clicked.connect(self.on_save_nexus)

        # RXES controls
        self.rxes_panel.rb_upper.toggled.connect(self.on_rxes_channel_changed)
        self.rxes_panel.rb_lower.toggled.connect(self.on_rxes_channel_changed)
        self.rxes_panel.rb_mode_incident.toggled.connect(self.on_mode_changed)
        self.rxes_panel.rb_mode_transfer.toggled.connect(self.on_mode_changed)
        self.rxes_panel.btn_load_xes.clicked.connect(self.on_rxes_normalise)
        self.rxes_panel.btn_add_line.clicked.connect(self.on_add_line)
        self.rxes_panel.btn_remove_line.clicked.connect(self.on_remove_line)
        self.rxes_panel.btn_update_spectrum.clicked.connect(self.on_update_spectrum)
        self.rxes_panel.btn_save_spectrum.clicked.connect(self.on_save_spectrum)
        self.rxes_panel.sl_width.valueChanged.connect(self.on_bandwidth_changed)
        self.rxes_panel.rb_extr_incident.toggled.connect(self.on_extraction_changed)
        self.rxes_panel.rb_extr_emission.toggled.connect(self.on_extraction_changed)
        self.rxes_panel.rb_extr_transfer.toggled.connect(self.on_extraction_changed)

        # XES controls (multi-scan)
        self.xes_panel.btn_load_scans.clicked.connect(self.on_xes_load_scans)
        self.xes_panel.btn_remove_selected.clicked.connect(self.on_xes_remove_selected)
        self.xes_panel.btn_clear_all.clicked.connect(self.on_xes_clear_all)
        self.xes_panel.btn_average.clicked.connect(self.on_xes_average_selected)
        self.xes_panel.btn_save_avg_norm.clicked.connect(self.on_xes_save_normalised_average)
        self.xes_panel.btn_save_average.clicked.connect(self.on_xes_save_average)
        # Normalise Average section
        self.xes_panel.btn_load_xes.clicked.connect(self.on_xes_normalise_average)
        self.xes_panel.rb_upper.toggled.connect(self.update_status_label)
        self.xes_panel.rb_lower.toggled.connect(self.update_status_label)

        # Tabs change
        self.tabs.currentChanged.connect(self.on_tab_changed)

        # Status bar
        self.status = self.statusBar()
        self.status.showMessage("Ready")

        self.update_ui_state()
        self._refresh_xes_plot()

    # ---------------- UI state ----------------

    def update_ui_state(self):
        has_data = self.dataset is not None
        self.io_panel.btn_save_ascii.setEnabled(has_data)
        try:
            self.io_panel.btn_save_nexus.setEnabled(has_data and io_mod.H5_AVAILABLE)
            if not io_mod.H5_AVAILABLE:
                self.io_panel.btn_save_nexus.setToolTip("Install h5py to enable NeXus saving: pip install h5py")
            else:
                self.io_panel.btn_save_nexus.setToolTip("")
        except Exception:
            pass

        if self.dataset:
            self.io_panel.path_edit.setText(self.dataset.source or "")
            if self.dataset.kind == "1D":
                n = len(self.dataset.y) if self.dataset.y is not None else 0
                self.io_panel.info_label.setText(f"1D spectrum, N={n}")
            elif self.dataset.kind == "2D":
                ny, nx = self.dataset.z.shape
                self.io_panel.info_label.setText(f"2D map, shape={ny}×{nx}")
            else:
                self.io_panel.info_label.setText("Unknown dataset")
        else:
            self.io_panel.path_edit.setText("")
            self.io_panel.info_label.setText("No data loaded")

        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        self.rxes_panel.rb_extr_incident.setEnabled(True)
        self.rxes_panel.rb_extr_emission.setEnabled(incident_mode)
        self.rxes_panel.rb_extr_transfer.setEnabled(not incident_mode)

        if self.tabs.currentIndex() == 1 or (self.dataset and self.dataset.kind == "1D"):
            chan = "Upper" if self.xes_panel.rb_upper.isChecked() else "Lower"
            mode = "XES (1D)"
        else:
            chan = "Upper" if self.rxes_panel.rb_upper.isChecked() else "Lower"
            mode = "Incident Energy (Ω vs ω)" if incident_mode else "Energy Transfer (Ω vs Ω−ω)"
        base = os.path.basename(self.dataset.source) if self.dataset and self.dataset.source else ""
        self.io_panel.status_label.setText(f"Channel: {chan} | Mode: {mode} | File: {base}")

        if self.current_scan_number is not None and self.current_scan_number in self.scan:
            nf = self.scan[self.current_scan_number].get("norm_factor", None)
            if nf and np.isfinite(nf) and nf > 0:
                self.rxes_panel.lbl_norm.setText(f"Normalised by XES area: {nf:.6g}")
            else:
                self.rxes_panel.lbl_norm.setText("No normalisation applied")
        else:
            self.rxes_panel.lbl_norm.setText("No normalisation applied")

    def set_dataset(self, dataset: Optional[DataSet]):
        self.dataset = dataset
        self.plot.plot(self.dataset)
        if self.dataset and self.dataset.kind == "2D":
            try:
                self.plot.set_signal_suppressed(True)
                self.set_line_orientation_for_current_mode()
                self.plot.ensure_line_count(1)
                self.plot.set_bandwidth(self._get_bandwidth_ev())
                self.plot.set_signal_suppressed(False)
                self.update_profiles()
            except Exception:
                pass
        self.update_ui_state()

    # ---------------- Load dispatcher ----------------

    def on_load(self):
        choices = ["RXES scan (.nxs)", "XES spectra (1D, multiple)"]
        choice, ok = QInputDialog.getItem(self, "Load", "What would you like to load?", choices, 0, False)
        if not ok:
            return
        if choice.startswith("RXES"):
            self._load_rxes_scan()
        else:
            self.on_xes_load_scans()

    def _load_rxes_scan(self):
        filters = ["NeXus scans (*.nxs)", "All files (*)"]
        path, _ = QFileDialog.getOpenFileName(self, "Load RXES scan", "", ";;".join(filters))
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".h5", ".hdf", ".hdf5") and i20_loader.is_probably_detector_hdf(path):
                QMessageBox.warning(self, "Detector HDF selected",
                                    "This looks like a raw detector file. Please pick the scan .nxs.")
                return
            scan_number = i20_loader.add_scan_from_nxs(self.scan, path)
            self.current_scan_number = scan_number
            self.status.showMessage(f"Loaded RXES: {path}", 5000)
            self.tabs.setCurrentIndex(0)
            self.refresh_rxes_view()
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load RXES:\n{path}\n\n{e}")

    # ---------------- XES: multi-scan workflow ----------------

    def on_xes_load_scans(self):
        filters = [
            "XES spectrum (*.nxs *.txt *.dat *.csv)",
            "NeXus (*.nxs)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        paths, _ = QFileDialog.getOpenFileNames(self, "Load XES scans", "", ";;".join(filters))
        if not paths:
            return
        use_upper = self.xes_panel.rb_upper.isChecked()
        channel = "upper" if use_upper else "lower"

        added = 0
        for path in paths:
            try:
                x, y = i20_loader.xes_from_path(path, channel=channel, type="XES")
                order = np.argsort(x)
                x = np.asarray(x)[order]; y = np.asarray(y)[order]
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                self._xes_items.append({
                    "path": path, "channel": channel, "x": x, "y": y, "label": os.path.basename(path)
                })
                self.xes_panel.add_item(os.path.basename(path), checked=True)
                added += 1
            except Exception as e:
                QMessageBox.warning(self, "Load XES", f"Failed to load {path}:\n{e}")

        if added:
            self.status.showMessage(f"Loaded {added} XES scans", 5000)
            self._xes_avg = None
            self._xes_avg_norm_factor = None
            self.xes_panel.lbl_norm.setText("Average: no normalisation")
            self._refresh_xes_plot()
            self.update_status_label()

    def on_xes_remove_selected(self):
        selected_rows = sorted({i.row() for i in self.xes_panel.list.selectedIndexes()}, reverse=True)
        for r in selected_rows:
            if 0 <= r < len(self._xes_items):
                self._xes_items.pop(r)
            try:
                self.xes_panel.list.takeItem(r)
            except Exception:
                pass
        self._xes_avg = None
        self._xes_avg_norm_factor = None
        self.xes_panel.lbl_norm.setText("Average: no normalisation")
        self._refresh_xes_plot()
        self.update_status_label()
        self._update_xes_buttons()

    def on_xes_clear_all(self):
        self._xes_items.clear()
        self._xes_avg = None
        self._xes_avg_norm_factor = None
        self.xes_panel.clear_items()
        self.xes_panel.lbl_norm.setText("Average: no normalisation")
        self._refresh_xes_plot()
        self.update_status_label()
        self._update_xes_buttons()

    def on_xes_average_selected(self):
        idxs = self.xes_panel.checked_indices()
        idxs = [i for i in idxs if 0 <= i < len(self._xes_items)]
        if not idxs:
            QMessageBox.information(self, "Average XES", "No scans ticked for averaging.")
            return
        xs = [self._xes_items[i]["x"] for i in idxs]
        ys = [self._xes_items[i]["y"] for i in idxs]
        xt, yt = self._regrid_and_average(xs, ys)
        if xt.size == 0:
            QMessageBox.warning(self, "Average", "No overlapping domain found to average.")
            return
        self._xes_avg = (xt, yt)
        self._xes_avg_norm_factor = None
        self.xes_panel.lbl_norm.setText("Average: no normalisation")
        self.dataset = DataSet("1D", x=xt, y=yt, xlabel="ω (eV)", ylabel="Intensity (XES)", source="")
        self._refresh_xes_plot()
        self.update_status_label()
        self.status.showMessage(f"Averaged {len(idxs)} scan(s)", 4000)
        self._update_xes_buttons()

    def _regrid_and_average(self, x_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if len(x_list) == 1:
            return x_list[0], y_list[0]
        try:
            lo = max([np.nanmin(x) for x in x_list if x.size])
            hi = min([np.nanmax(x) for x in x_list if x.size])
        except Exception:
            return np.array([]), np.array([])

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            x0 = x_list[0]
            ys = [np.interp(x0, x, y, left=np.nan, right=np.nan) for x, y in zip(x_list, y_list)]
            Y = np.nanmean(np.vstack(ys), axis=0)
            ok = np.isfinite(x0) & np.isfinite(Y)
            return x0[ok], Y[ok]

        x0 = x_list[0]
        dx = np.diff(x0)
        step = float(np.nanmedian(dx[dx > 0])) if dx.size else max(0.1, (hi - lo) / 1000.0)
        xt = np.arange(lo, hi + 0.5 * step, step)
        Ys = [np.interp(xt, x, y, left=np.nan, right=np.nan) for x, y in zip(x_list, y_list)]
        Yt = np.nanmean(np.vstack(Ys), axis=0)
        ok = np.isfinite(xt) & np.isfinite(Yt)
        return xt[ok], Yt[ok]

    # ---------------- XES: Normalise Average (like RXES normalise) ----------------

    def on_xes_normalise_average(self):
        """
        Use an external XES spectrum to select an area and normalise:
        - If an average exists, normalise the average.
        - Else if only one scan is loaded, normalise that scan.
        - Else prompt to average first.
        """
        # Decide what to normalise (average if present, otherwise single)
        target = None
        if self._xes_avg is not None and self._xes_avg[0].size and self._xes_avg[1].size:
            target = ("avg", (self._xes_avg[0], self._xes_avg[1]))
        elif len(self._xes_items) == 1:
            target = ("single", 0)

        if target is None:
            QMessageBox.information(self, "Normalise XES", "Average first (or load a single scan).")
            return

        # 1) Pick an external XES file
        filters = [
            "XES spectrum (*.nxs *.txt *.dat *.csv)",
            "NeXus (*.nxs)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        path, _ = QFileDialog.getOpenFileName(self, "Load XES spectrum for normalisation", "", ";;".join(filters))
        if not path:
            return

        try:
            # 2) Load it using the selected channel (Upper/Lower)
            use_upper = self.xes_panel.rb_upper.isChecked()
            channel = "upper" if use_upper else "lower"
            x_ext, y_ext = i20_loader.xes_from_path(path, channel=channel, type="XES")

            # 3) Let the user select an area on that external spectrum
            dlg = NormaliseDialog(x_ext, y_ext, parent=self, title=f"Select area (XES {'Upper' if use_upper else 'Lower'})")
            if dlg.exec() != QDialog.Accepted:
                return
            area = dlg.selected_area()
            if not np.isfinite(area) or area <= 0:
                QMessageBox.warning(self, "Normalise XES", "Selected area is invalid or non‑positive.")
                return

            # 4) Apply normalisation to the target
            if target[0] == "avg":
                x, y = target[1]
                y_norm = y / area
                self._xes_avg = (x, y_norm)
                self._xes_avg_norm_factor = float(area)
                self.dataset = DataSet("1D", x=x, y=y_norm, xlabel="ω (eV)", ylabel="Intensity / area", source="")
                self.xes_panel.lbl_norm.setText(f"Average normalised by area: {area:.6g}")
            else:
                idx = target[1]
                item = self._xes_items[idx]
                y_norm = item["y"] / area
                self._xes_items[idx] = {**item, "y": y_norm}
                self._xes_avg = None
                self._xes_avg_norm_factor = float(area)
                self.dataset = DataSet("1D", x=item["x"], y=y_norm, xlabel="ω (eV)", ylabel="Intensity / area", source=item["path"])
                self.xes_panel.lbl_norm.setText(f"Scan normalised by area: {area:.6g}")

            # 5) Refresh the overlays/average plot
            self._refresh_xes_plot()
            self.update_status_label()
            self.status.showMessage("XES normalisation applied", 4000)
            self._update_xes_buttons()

        except Exception as e:
            QMessageBox.critical(self, "XES normalise error", f"Failed to normalise XES:\n{path}\n\n{e}")

    # ---------------- XES: save ----------------

    def on_xes_save_average(self):
        if self._xes_avg is None:
            QMessageBox.information(self, "Save average", "No averaged spectrum to save.")
            return
        x, y = self._xes_avg
        default_name = "xes_average.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save XES average", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            arr = np.column_stack([x, y])
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("omega_eV,intensity\n")
                np.savetxt(fh, arr, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved XES average: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save average:\n{e}")

    def _update_xes_buttons(self):
        has_avg = self._xes_avg is not None and self._xes_avg[0].size > 0
        self.xes_panel.btn_save_average.setEnabled(has_avg)
        has_norm = has_avg and bool(self._xes_avg_norm_factor)
        self.xes_panel.btn_save_avg_norm.setEnabled(has_norm)

    # Implement the save-normalised-average action:
    def on_xes_save_normalised_average(self):
        """
        Save the normalised average (requires that a normalisation factor has been applied).
        """
        if self._xes_avg is None or not self._xes_avg_norm_factor:
            QMessageBox.information(self, "Save normalised average", "No normalised average to save.")
            return
        x, y = self._xes_avg
        default_name = "xes_average_normalised.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save XES normalised average", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            arr = np.column_stack([x, y])
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("omega_eV,intensity_normalised\n")
                np.savetxt(fh, arr, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved XES normalised average: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save normalised average:\n{e}")

    # ---------------- Helpers: plotting/status ----------------

    def _refresh_xes_plot(self):
        curves = [{"x": it["x"], "y": it["y"], "label": it["label"], "color": None, "alpha": 0.7}
                  for it in self._xes_items]
        avg = {"x": self._xes_avg[0], "y": self._xes_avg[1], "label": "Average (XES)"} if self._xes_avg is not None else None
        try:
            self.plot.plot_xes_bundle(curves, avg=avg, title="XES scans (overlays)")
        except Exception:
            self.plot.plot(self.dataset if self.dataset else None)

        if avg is not None:
            self.dataset = DataSet("1D", x=avg["x"], y=avg["y"], xlabel="ω (eV)", ylabel="Intensity (XES)")
        self.update_status_label()

    def update_status_label(self):
        if self.tabs.currentIndex() == 1 or (self.dataset and self.dataset.kind == "1D"):
            chan = "Upper" if self.xes_panel.rb_upper.isChecked() else "Lower"
            base = ""
            self.io_panel.status_label.setText(f"Channel: {chan} | Mode: XES (1D bundle) | File: {base}")
        else:
            self.update_ui_state()
   
    # ---------------- RXES view build ----------------

    def refresh_rxes_view(self):
        if self.current_scan_number is None:
            return
        entry = self.scan.get(self.current_scan_number)
        if not entry:
            return

        use_upper = self.rxes_panel.rb_upper.isChecked()
        mode_transfer = self.rxes_panel.rb_mode_transfer.isChecked()

        # Select emission/intensity by channel
        if use_upper:
            emission_2d = entry.get("energy_upper_2d")
            Z = entry.get("intensity_upper")
            zlabel = "FFI1_medipix1 (counts)"
        else:
            emission_2d = entry.get("energy_lower_2d")
            Z = entry.get("intensity_lower")
            zlabel = "FFI1_medipix2 (counts)"

        if emission_2d is None or Z is None:
            QMessageBox.warning(self, "Missing channel", "Selected channel not present in this scan.")
            return

        bragg_off_2d = entry.get("braggOffset_2d")
        if bragg_off_2d is None:
            QMessageBox.critical(self, "Missing axis", "bragg1WithOffset grid not found.")
            return

        # Axes: Y = ω (rows), X = Ω (cols)
        y_omega, x_Omega, transposed = i20_loader.reduce_axes_for(emission_2d, bragg_off_2d)
        Z = np.asarray(Z, dtype=float)
        if transposed:
            Z = Z.T

        # Apply RXES normalisation factor (by XES area) if present
        nf = entry.get("norm_factor", None)
        if nf and np.isfinite(nf) and nf > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                Z = Z / nf
            if "area" not in zlabel:
                zlabel = zlabel + " / area"

        # Build DataSet for plotting
        if mode_transfer:
            # Energy transfer view: X=Ω, Y=Ω−ω
            X2D = np.broadcast_to(x_Omega[None, :], Z.shape)
            Y2D = X2D - np.broadcast_to(y_omega[:, None], Z.shape)
            ds = DataSet("2D",
                         x2d=X2D, y2d=Y2D, z=Z,
                         xlabel="Ω (eV)", ylabel="Ω − ω (eV)", zlabel=zlabel,
                         source=entry.get("path", ""))
        else:
            ds = DataSet("2D",
                         x=x_Omega, y=y_omega, z=Z,
                         xlabel="Ω (eV)", ylabel="ω (eV)", zlabel=zlabel,
                         source=entry.get("path", ""))

        self.set_dataset(ds)
        try:
            self.plot.autoscale_current()
        except Exception:
            pass

    # ---------------- Mode and extraction ----------------

    def on_mode_changed(self):
        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        if incident_mode and self.rxes_panel.rb_extr_transfer.isChecked():
            self.rxes_panel.rb_extr_incident.setChecked(True)
        if not incident_mode and self.rxes_panel.rb_extr_emission.isChecked():
            self.rxes_panel.rb_extr_incident.setChecked(True)
        self.refresh_rxes_view()
        try:
            self.plot.autoscale_current()
        except Exception:
            pass

    def on_extraction_changed(self):
        # If user selects Constant Transfer while in Incident mode, switch modes
        if self.rxes_panel.rb_extr_transfer.isChecked() and self.rxes_panel.rb_mode_incident.isChecked():
            self.rxes_panel.rb_mode_transfer.setChecked(True)
            return
        try:
            self.plot.set_signal_suppressed(True)
            self.set_line_orientation_for_current_mode()
            self.plot.set_signal_suppressed(False)
        except Exception:
            pass
        try:
            self.plot.autoscale_current()
        except Exception:
            pass
        self.update_profiles()

    def set_line_orientation_for_current_mode(self):
        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        if incident_mode:
            # Incident view (Ω vs ω)
            if self.rxes_panel.rb_extr_emission.isChecked():
                self.plot.set_line_orientation("horizontal")
            else:
                self.plot.set_line_orientation("vertical")
        else:
            # Transfer view (Ω vs Ω−ω)
            if self.rxes_panel.rb_extr_transfer.isChecked():
                self.plot.set_line_orientation("horizontal")
            else:
                self.plot.set_line_orientation("vertical")

    # ---------------- Channel switching ----------------

    def on_rxes_channel_changed(self):
        # Rebuild RXES view if a scan is loaded
        if self.current_scan_number is not None:
            self.refresh_rxes_view()
        self.update_ui_state()

    def on_tab_changed(self, idx: int):
        self.update_ui_state()

    # ---------------- ROI/profiles ----------------

    def _build_coordinate_grids(self, ds: DataSet) -> Tuple[np.ndarray, np.ndarray]:
        Z = ds.z
        if ds.x2d is not None and ds.y2d is not None:
            return ds.x2d, ds.y2d
        elif ds.x is not None and ds.y is not None:
            X, Y = np.meshgrid(ds.x, ds.y)  # (ny, nx)
            return X, Y
        else:
            ny, nx = Z.shape
            X = np.arange(nx, dtype=float)[None, :].repeat(ny, axis=0)
            Y = np.arange(ny, dtype=float)[:, None].repeat(nx, axis=1)
            return X, Y

    def on_add_line(self):
        try:
            self.plot.add_line()
        except Exception:
            pass
        self.update_profiles()

    def on_remove_line(self):
        try:
            self.plot.remove_line()
        except Exception:
            pass
        self.update_profiles()

    def on_update_spectrum(self):
        self.update_profiles()

    def on_save_spectrum(self):
        # Ensure latest profiles
        self.update_profiles()
        if not self._last_profiles:
            QMessageBox.information(self, "No data", "No spectra to save. Add/move lines to generate profiles.")
            return
        # Build combined array as [X1,Y1,X2,Y2,...]
        maxlen = max(len(p[0]) for p in self._last_profiles)
        cols = []
        headers = []
        for (xv, yv, lbl) in self._last_profiles:
            pad = maxlen - len(xv)
            if pad > 0:
                xv = np.pad(xv, (0, pad), constant_values=np.nan)
                yv = np.pad(yv, (0, pad), constant_values=np.nan)
            cols.extend([xv, yv])
            headers.extend([f"X {lbl}", f"Y {lbl}"])
        out = np.column_stack(cols)
        default_name = "profiles.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save spectra as CSV", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(", ".join(headers) + "\n")
                np.savetxt(fh, out, delimiter=",", comments="", fmt="%.10g")
            self.status.showMessage(f"Saved spectra: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save spectra:\n{path}\n\n{e}")

    def on_bandwidth_changed(self, v: int):
        width_ev = self._get_bandwidth_ev()
        try:
            self.plot.set_bandwidth(width_ev)
        except Exception:
            pass
        self.update_profiles()

    def _get_bandwidth_ev(self) -> float:
        # Preferred path: let the panel do the scaling
        if hasattr(self.rxes_panel, "bandwidth_ev"):
            try:
                return float(self.rxes_panel.bandwidth_ev())
            except Exception:
                pass
        # Fallback: infer scaling from slider range
        try:
            sl = self.rxes_panel.sl_width
            v = float(sl.value())
            # If the slider max is > 3, it's the scaled 0.2 eV-step slider (1..15) -> divide by 5
            return v / 5.0 if sl.maximum() > 3 else v
        except Exception:
            return 1.0

    def update_profiles(self):
        self._last_profiles = []
        if self.dataset is None or self.dataset.kind != "2D":
            return

        ds = self.dataset
        Z = np.asarray(ds.z, dtype=float)
        X2D, Y2D = self._build_coordinate_grids(ds)

        incident_view = (ds.x2d is None or ds.y2d is None)
        width_ev = self._get_bandwidth_ev()

        line_vals = []
        try:
            line_vals = self.plot.get_line_positions()
        except Exception:
            pass
        if len(line_vals) == 0:
            try:
                self.plot.plot_profiles("", [])
            except Exception:
                pass
            self.update_ui_state()
            return

        curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
        if self.rxes_panel.rb_extr_incident.isChecked():
            if incident_view:
                x_profile = ds.y if ds.y is not None else Y2D[:, 0]
                for val in line_vals:
                    mask = np.abs(X2D - val) <= (width_ev / 2.0)
                    y_profile = np.nansum(np.where(mask, Z, 0.0), axis=1)
                    curves.append((x_profile, y_profile, f"Ω={val:.2f} eV"))
                xlab = "ω (eV)"
            else:
                x_profile = np.nanmedian(Y2D, axis=1)  # Δ = Ω−ω
                for val in line_vals:
                    mask = np.abs(X2D - val) <= (width_ev / 2.0)
                    y_profile = np.nansum(np.where(mask, Z, 0.0), axis=1)
                    curves.append((x_profile, y_profile, f"Ω={val:.2f} eV"))
                xlab = "Ω − ω (eV)"
            try:
                self.plot.plot_profiles(xlab, curves)
            except Exception:
                pass
            self._last_profiles = curves

        elif self.rxes_panel.rb_extr_emission.isChecked():
            if not incident_view:
                return
            x_profile = ds.x if ds.x is not None else X2D[0, :]
            for val in line_vals:
                mask = np.abs(Y2D - val) <= (width_ev / 2.0)
                y_profile = np.nansum(np.where(mask, Z, 0.0), axis=0)
                curves.append((x_profile, y_profile, f"ω={val:.2f} eV"))
            try:
                self.plot.plot_profiles("Ω (eV)", curves)
            except Exception:
                pass
            self._last_profiles = curves

        else:
            # Constant Transfer (Ω−ω): transfer view only
            if incident_view:
                self.rxes_panel.rb_mode_transfer.setChecked(True)
                return
            x_profile = ds.x2d[0, :] if ds.x2d is not None else X2D[0, :]
            for val in line_vals:
                mask = np.abs(Y2D - val) <= (width_ev / 2.0)
                y_profile = np.nansum(np.where(mask, Z, 0.0), axis=0)
                curves.append((x_profile, y_profile, f"Δ={val:.2f} eV"))
            try:
                self.plot.plot_profiles("Ω (eV)", curves)
            except Exception:
                pass
            self._last_profiles = curves

        self.update_ui_state()

    # ---------------- Normalisation ----------------

    def on_rxes_normalise(self):
        """
        Load a 1D XES from file and normalise current RXES map by selected area.
        """
        if self.current_scan_number is None:
            QMessageBox.information(self, "RXES", "Load an RXES .nxs scan first.")
            return
        filters = ["XES spectrum (*.nxs *.txt *.dat *.csv)", "NeXus (*.nxs)", "ASCII (*.txt *.dat *.csv)", "All files (*)"]
        path, _ = QFileDialog.getOpenFileName(self, "Load XES spectrum", "", ";;".join(filters))
        if not path:
            return
        try:
            use_upper = self.rxes_panel.rb_upper.isChecked()
            channel = "upper" if use_upper else "lower"
            x, y = i20_loader.xes_from_path(path, channel=channel, type="RXES")
            dlg = NormaliseDialog(x, y, parent=self, title=f"XES ({'Upper' if use_upper else 'Lower'})")
            if dlg.exec() != QDialog.Accepted:
                return
            area = dlg.selected_area()
            if not np.isfinite(area) or area <= 0:
                QMessageBox.warning(self, "Invalid area", "Selected XES area is not valid or non‑positive.")
                return
            self.scan[self.current_scan_number]["norm_factor"] = float(area)
            self.rxes_panel.lbl_norm.setText(f"Normalised by XES area: {area:.6g}")
            self.refresh_rxes_view()
        except Exception as e:
            QMessageBox.critical(self, "XES load error", f"Failed to load/normalise from XES:\n{path}\n\n{e}")

    # ---------------- XES: multi-scan workflow (steps 1–3) ----------------

    def on_xes_load_scans(self):
        """
        Load multiple XES scans (ASCII or .nxs) using selected channel, and add to bundle list.
        """
        filters = [
            "XES spectrum (*.nxs *.txt *.dat *.csv)",
            "NeXus (*.nxs)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        paths, _ = QFileDialog.getOpenFileNames(self, "Load XES scans", "", ";;".join(filters))
        if not paths:
            return

        use_upper = self.xes_panel.rb_upper.isChecked()
        channel = "upper" if use_upper else "lower"

        added = 0
        for path in paths:
            try:
                x, y = i20_loader.xes_from_path(path, channel=channel, type="XES")
                order = np.argsort(x)
                x = np.asarray(x)[order]; y = np.asarray(y)[order]
                ok = np.isfinite(x) & np.isfinite(y)
                x, y = x[ok], y[ok]
                self._xes_items.append({
                    "path": path,
                    "channel": channel,
                    "x": x,
                    "y": y,
                    "label": os.path.basename(path)
                })
                self.xes_panel.add_item(os.path.basename(path), checked=True)
                added += 1
            except Exception as e:
                QMessageBox.warning(self, "Load XES", f"Failed to load {path}:\n{e}")

        if added:
            self.status.showMessage(f"Loaded {added} XES scans", 5000)
            self._xes_avg = None
            self._xes_avg_norm_factor = None
            self.xes_panel.lbl_norm.setText("Average: no normalisation")
            self._refresh_xes_plot()
            self.update_status_label()
            self._update_xes_buttons()

    def on_xes_remove_selected(self):
        selected_rows = sorted({i.row() for i in self.xes_panel.list.selectedIndexes()}, reverse=True)
        for r in selected_rows:
            if 0 <= r < len(self._xes_items):
                self._xes_items.pop(r)
            try:
                self.xes_panel.list.takeItem(r)
            except Exception:
                pass
        self._xes_avg = None
        self._xes_avg_norm_factor = None
        self.xes_panel.lbl_norm.setText("Average: no normalisation")
        self._refresh_xes_plot()
        self.update_status_label()
        self._update_xes_buttons()

    def on_xes_clear_all(self):
        self._xes_items.clear()
        self._xes_avg = None
        self._xes_avg_norm_factor = None
        self.xes_panel.clear_items()
        self.xes_panel.lbl_norm.setText("Average: no normalisation")
        self._refresh_xes_plot()
        self.update_status_label()
        self._update_xes_buttons()

    def on_xes_average_selected(self):
        idxs = self.xes_panel.checked_indices()
        if not idxs:
            QMessageBox.information(self, "Average XES", "No scans ticked for averaging.")
            return

        # Filter any indices that are out of range (defensive)
        idxs = [i for i in idxs if 0 <= i < len(self._xes_items)]
        if not idxs:
            QMessageBox.information(self, "Average XES", "No valid scans found to average.")
            return

        xs = [self._xes_items[i]["x"] for i in idxs]
        ys = [self._xes_items[i]["y"] for i in idxs]

        xt, yt = self._regrid_and_average(xs, ys)
        if xt.size == 0:
            QMessageBox.warning(self, "Average", "No overlapping domain found to average.")
            return

        self._xes_avg = (xt, yt)
        self._xes_avg_norm_factor = None
        self.xes_panel.lbl_norm.setText("Average: no normalisation")

        # Also store as current dataset for saving (Save as ASCII)
        self.dataset = DataSet("1D", x=xt, y=yt, xlabel="ω (eV)", ylabel="Intensity (XES)", source="")
        self._refresh_xes_plot()
        self._update_xes_buttons()
        self.update_status_label()
        self.status.showMessage(f"Averaged {len(idxs)} scan(s)", 4000)

    def _regrid_and_average(self, x_list: List[np.ndarray], y_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regrid multiple XES curves to a common ω axis and average them.
        Strategy:
          - Compute overlapping ω-range across all scans.
          - Build a target grid using median step from the first scan within overlap.
          - Interpolate each scan onto the target grid; average ignoring NaNs.
        """
        if len(x_list) == 1:
            return x_list[0], y_list[0]

        # Overlap range
        try:
            lo = max([np.nanmin(x) for x in x_list if x.size])
            hi = min([np.nanmax(x) for x in x_list if x.size])
        except Exception:
            return np.array([]), np.array([])

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # Fallback: union grid of first scan
            x0 = x_list[0]
            ys = []
            for x, y in zip(x_list, y_list):
                yi = np.interp(x0, x, y, left=np.nan, right=np.nan)
                ys.append(yi)
            Y = np.nanmean(np.vstack(ys), axis=0)
            ok = np.isfinite(x0) & np.isfinite(Y)
            return x0[ok], Y[ok]

        # Target grid
        x0 = x_list[0]
        dx = np.diff(x0)
        step = float(np.nanmedian(dx[dx > 0])) if dx.size else max(0.1, (hi - lo) / 1000.0)
        xt = np.arange(lo, hi + 0.5 * step, step)

        # Regrid and average
        Ys = []
        for x, y in zip(x_list, y_list):
            yi = np.interp(xt, x, y, left=np.nan, right=np.nan)
            Ys.append(yi)
        Yt = np.nanmean(np.vstack(Ys), axis=0)
        ok = np.isfinite(xt) & np.isfinite(Yt)
        return xt[ok], Yt[ok]

    # def on_xes_normalise_average(self):
    #     if self._xes_avg is None or self._xes_avg[0].size == 0:
    #         QMessageBox.information(self, "Normalise XES", "Average first (Average selected).")
    #         return
    #     x, y = self._xes_avg
    #     dlg = NormaliseDialog(x, y, parent=self, title="Normalise XES average by area")
    #     if dlg.exec() != QDialog.Accepted:
    #         return
    #     area = dlg.selected_area()
    #     if not np.isfinite(area) or area <= 0:
    #         QMessageBox.warning(self, "Normalise XES", "Selected area is invalid or non‑positive.")
    #         return
    #     y_norm = y / area
    #     self._xes_avg = (x, y_norm)
    #     self._xes_avg_norm_factor = float(area)
    #     self.xes_panel.lbl_norm.setText(f"Average normalised by area: {area:.6g}")

    #     # Update dataset
    #     self.dataset = DataSet("1D", x=x, y=y_norm, xlabel="ω (eV)", ylabel="Intensity / area", source="")
    #     self._refresh_xes_plot()
    #     self.update_status_label()

    def on_xes_save_average(self):
        if self._xes_avg is None:
            QMessageBox.information(self, "Save average", "No averaged spectrum to save.")
            return
        x, y = self._xes_avg
        default_name = "xes_average.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save XES average", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            arr = np.column_stack([x, y])
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write("omega_eV,intensity\n")
                np.savetxt(fh, arr, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved XES average: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save average:\n{e}")

    def on_xes_save_overlays(self):
        if not self._xes_items:
            QMessageBox.information(self, "Save overlays", "No XES scans loaded.")
            return
        maxlen = max(len(it["x"]) for it in self._xes_items)
        cols = []
        headers = []
        for it in self._xes_items:
            x = it["x"]; y = it["y"]
            pad = maxlen - len(x)
            if pad > 0:
                x = np.pad(x, (0, pad), constant_values=np.nan)
                y = np.pad(y, (0, pad), constant_values=np.nan)
            cols.extend([x, y])
            headers.extend([f"X {it['label']}", f"Y {it['label']}"])
        out = np.column_stack(cols)
        default_name = "xes_overlays.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Save XES overlays", default_name, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(", ".join(headers) + "\n")
                np.savetxt(fh, out, delimiter=",", fmt="%.10g", comments="")
            self.status.showMessage(f"Saved XES overlays: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save overlays:\n{e}")

    def _refresh_xes_plot(self):
        # Build curves list for overlay
        curves = []
        for item in self._xes_items:
            curves.append({
                "x": item["x"],
                "y": item["y"],
                "label": item["label"],
                "color": None,
                "alpha": 0.7
            })

        avg = None
        if self._xes_avg is not None:
            avg = {"x": self._xes_avg[0], "y": self._xes_avg[1], "label": "Average (XES)"}

        try:
            self.plot.plot_xes_bundle(curves, avg=avg, title="XES scans (overlays)")
        except Exception:
            # Fallback to simple plot if helper not present
            self.plot.plot(self.dataset if self.dataset else None)

        # Update I/O info panel for XES
        if avg is not None:
            # Also set dataset to the average so Save as ASCII works
            self.dataset = DataSet("1D", x=avg["x"], y=avg["y"], xlabel="ω (eV)", ylabel="Intensity (XES)")
        else:
            # Do not clear dataset here; Save as ASCII will be disabled based on has_data
            pass

        self.update_status_label()
        self._update_xes_buttons()

    def update_status_label(self):
        # Status for XES tab
        if self.tabs.currentIndex() == 1 or (self.dataset and self.dataset.kind == "1D"):
            chan = "Upper" if self.xes_panel.rb_upper.isChecked() else "Lower"
            base = ""  # multiple files: no single name
            self.io_panel.status_label.setText(f"Channel: {chan} | Mode: XES (1D bundle) | File: {base}")
        else:
            self.update_ui_state()

    # ---------------- Generic saves ----------------

    def on_save_ascii(self):
        if self.dataset is None:
            QMessageBox.information(self, "No data", "Nothing to save. Load/compute a dataset first.")
            return
        base = "spectrum.csv" if self.dataset.kind == "1D" else "data.csv"
        start = os.path.join(os.path.dirname(self.dataset.source), base) if self.dataset.source else base
        path, _ = QFileDialog.getSaveFileName(self, "Save as ASCII", start, "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            io_mod.save_ascii(path, self.dataset)
            self.status.showMessage(f"Saved ASCII: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save ASCII:\n{path}\n\n{e}")

    def on_save_nexus(self):
        if self.dataset is None:
            QMessageBox.information(self, "No data", "Nothing to save.")
            return
        try:
            if not io_mod.H5_AVAILABLE:
                QMessageBox.warning(self, "h5py missing", "Install h5py to save NeXus files: pip install h5py")
                return
        except Exception:
            pass
        base = "data.nxs"
        start = os.path.join(os.path.dirname(self.dataset.source), base) if self.dataset.source else base
        path, _ = QFileDialog.getSaveFileName(self, "Save as NeXus (HDF5)", start, "NeXus/HDF5 (*.nxs *.h5 *.hdf5);;All files (*)")
        if not path:
            return
        try:
            io_mod.save_nexus(path, self.dataset)
            self.status.showMessage(f"Saved NeXus: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save NeXus:\n{path}\n\n{e}")