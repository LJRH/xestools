import os
from typing import Optional, List, Tuple
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QDialog, QMessageBox
)

from i20_xes.modules.dataset import DataSet
from i20_xes.widgets.io_panel import IOPanel, RXESPanel
from i20_xes.widgets.plot_widget import PlotWidget
from i20_xes.widgets.normalise_dialog import NormaliseDialog
from i20_xes.modules import io as io_mod
from i20_xes.modules.scan import Scan
from i20_xes.modules import i20_loader


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("I20 XES/RXES Viewer")
        self.resize(1300, 880)

        self.scan = Scan()
        self.current_scan_number: Optional[int] = None
        self.dataset: Optional[DataSet] = None
        self._last_profiles: List[Tuple[np.ndarray, np.ndarray, str]] = []

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # Left panels
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)

        self.io_panel = IOPanel()
        self.rxes_panel = RXESPanel()

        left_layout.addWidget(self.io_panel)
        left_layout.addWidget(self.rxes_panel)
        left_layout.addStretch(1)

        # Right plot
        self.plot = PlotWidget()
        self.plot.lines_changed.connect(self.update_profiles)

        root_layout.addWidget(left_container, 0)
        root_layout.addWidget(self.plot, 1)

        # I/O actions
        self.io_panel.btn_load.clicked.connect(self.on_load)
        self.io_panel.btn_save_ascii.clicked.connect(self.on_save_ascii)
        self.io_panel.btn_save_nexus.clicked.connect(self.on_save_nexus)

        # RXES controls
        self.rxes_panel.rb_upper.toggled.connect(self.refresh_rxes_view)
        self.rxes_panel.rb_lower.toggled.connect(self.refresh_rxes_view)
        self.rxes_panel.rb_mode_incident.toggled.connect(self.on_mode_changed)
        self.rxes_panel.rb_mode_transfer.toggled.connect(self.on_mode_changed)

        # Normalisation
        self.rxes_panel.btn_load_xes.clicked.connect(self.on_load_xes)

        # ROI controls
        self.rxes_panel.btn_add_line.clicked.connect(self.on_add_line)
        self.rxes_panel.btn_remove_line.clicked.connect(self.on_remove_line)
        self.rxes_panel.btn_update_spectrum.clicked.connect(self.on_update_spectrum)
        self.rxes_panel.btn_save_spectrum.clicked.connect(self.on_save_spectrum)
        self.rxes_panel.sl_width.valueChanged.connect(self.on_bandwidth_changed)
        self.rxes_panel.rb_extr_incident.toggled.connect(self.on_extraction_changed)
        self.rxes_panel.rb_extr_emission.toggled.connect(self.on_extraction_changed)
        self.rxes_panel.rb_extr_transfer.toggled.connect(self.on_extraction_changed)

        self.status = self.statusBar()
        self.status.showMessage("Ready")

        self.update_ui_state()

    # ----- UI state -----

    def update_ui_state(self):
        has_data = self.dataset is not None
        self.io_panel.btn_save_ascii.setEnabled(has_data)
        self.io_panel.btn_save_nexus.setEnabled(has_data and io_mod.H5_AVAILABLE)
        if not io_mod.H5_AVAILABLE:
            self.io_panel.btn_save_nexus.setToolTip("Install h5py to enable NeXus saving: pip install h5py")
        else:
            self.io_panel.btn_save_nexus.setToolTip("")

        if self.dataset:
            self.io_panel.path_edit.setText(self.dataset.source or "")
            if self.dataset.kind == "1D":
                n = len(self.dataset.y) if self.dataset.y is not None else 0
                self.io_panel.info_label.setText(f"1D spectrum, N={n}")
            else:
                ny, nx = self.dataset.z.shape
                self.io_panel.info_label.setText(f"2D map, shape={ny}x{nx}")
        else:
            self.io_panel.path_edit.clear()
            self.io_panel.info_label.setText("No data loaded")

        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        self.rxes_panel.rb_extr_incident.setEnabled(True)
        self.rxes_panel.rb_extr_emission.setEnabled(incident_mode)
        self.rxes_panel.rb_extr_transfer.setEnabled(not incident_mode)

        chan = "Upper" if self.rxes_panel.rb_upper.isChecked() else "Lower"
        mode = "Incident Energy (Ω vs ω)" if incident_mode else "Energy Transfer (Ω vs Ω−ω)"
        base = os.path.basename(self.dataset.source) if self.dataset and self.dataset.source else ""
        self.io_panel.status_label.setText(f"Channel: {chan} | Mode: {mode} | File: {base}")

        # Update normalisation label from current scan
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

        self.plot.set_signal_suppressed(True)
        self.set_line_orientation_for_current_mode()
        self.plot.ensure_line_count(1)
        self.plot.set_bandwidth(self.rxes_panel.bandwidth_ev())
        self.plot.set_signal_suppressed(False)

        self.update_ui_state()
        self.update_profiles()

    # ----- Load RXES data -----

    def on_load(self):
        filters = [
            "NeXus scans (*.nxs)",
            "NeXus/HDF5 (*.nxs *.nx5 *.h5 *.hdf *.hdf5)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        path, _ = QFileDialog.getOpenFileName(self, "Load data", "", ";;".join(filters))
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()

        if ext in (".h5", ".hdf", ".hdf5"):
            try:
                if i20_loader.is_probably_detector_hdf(path):
                    QMessageBox.warning(
                        self, "Detector HDF selected",
                        "This file looks like a detector data file (raw images).\n"
                        "Please select the corresponding scan .nxs file (e.g., 279517_1.nxs)."
                    )
                    return
            except Exception:
                pass

        if ext == ".nxs":
            try:
                scan_number = i20_loader.add_scan_from_nxs(self.scan, path)
                self.current_scan_number = scan_number
                self.status.showMessage(f"Loaded I20 scan: {path} -> scan {scan_number}", 5000)
                self.refresh_rxes_view()
                return
            except Exception as e:
                QMessageBox.critical(self, "Load error", f"Failed to load I20 scan (.nxs):\n{path}\n\n{e}")
                self.status.showMessage("Load failed", 5000)
                return

        try:
            ds = io_mod.load_path(path)
            self.set_dataset(ds)
            self.status.showMessage(f"Loaded: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Failed to load:\n{path}\n\n{e}")
            self.status.showMessage("Load failed", 5000)

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

        # Axes
        y_omega, x_Omega, transposed = i20_loader.reduce_axes_for(emission_2d, bragg_off_2d)
        Z = np.asarray(Z, dtype=float)
        if transposed:
            Z = Z.T

        # Apply normalisation factor if present
        nf = entry.get("norm_factor", None)
        if nf and np.isfinite(nf) and nf > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                Z = Z / nf
            zlabel = (zlabel + " / area") if "area" not in zlabel else zlabel

        # Build dataset
        if mode_transfer:
            X2D = np.broadcast_to(x_Omega[None, :], Z.shape)
            Y2D = X2D - np.broadcast_to(y_omega[:, None], Z.shape)
            ds = DataSet("2D", x2d=X2D, y2d=Y2D, z=Z, xlabel="Ω (eV)", ylabel="Ω − ω (eV)",
                         zlabel=zlabel, source=entry.get("path", ""))
        else:
            ds = DataSet("2D", x=x_Omega, y=y_omega, z=Z, xlabel="Ω (eV)", ylabel="ω (eV)",
                         zlabel=zlabel, source=entry.get("path", ""))
        self.set_dataset(ds)

    # ----- Mode / extraction -----

    def on_mode_changed(self):
        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        if incident_mode and self.rxes_panel.rb_extr_transfer.isChecked():
            self.rxes_panel.rb_extr_incident.setChecked(True)
        if not incident_mode and self.rxes_panel.rb_extr_emission.isChecked():
            self.rxes_panel.rb_extr_incident.setChecked(True)
        self.refresh_rxes_view()
        self.plot.autoscale_current()

    def on_extraction_changed(self):
        if self.rxes_panel.rb_extr_transfer.isChecked() and self.rxes_panel.rb_mode_incident.isChecked():
            self.rxes_panel.rb_mode_transfer.setChecked(True)
            return
        self.plot.set_signal_suppressed(True)
        self.set_line_orientation_for_current_mode()
        self.plot.set_signal_suppressed(False)
        self.plot.autoscale_current()
        self.update_profiles()

    def set_line_orientation_for_current_mode(self):
        incident_mode = self.rxes_panel.rb_mode_incident.isChecked()
        if incident_mode:
            if self.rxes_panel.rb_extr_emission.isChecked():
                self.plot.set_line_orientation("horizontal")
            else:
                self.plot.set_line_orientation("vertical")
        else:
            if self.rxes_panel.rb_extr_transfer.isChecked():
                self.plot.set_line_orientation("horizontal")
            else:
                self.plot.set_line_orientation("vertical")

    # ----- ROI / profiles (unchanged from previous working version except using bandwidth_ev) -----

    def _build_coordinate_grids(self, ds: DataSet) -> Tuple[np.ndarray, np.ndarray]:
        Z = ds.z
        if ds.x2d is not None and ds.y2d is not None:
            return ds.x2d, ds.y2d
        elif ds.x is not None and ds.y is not None:
            X, Y = np.meshgrid(ds.x, ds.y)
            return X, Y
        else:
            ny, nx = Z.shape
            X = np.arange(nx, dtype=float)[None, :].repeat(ny, axis=0)
            Y = np.arange(ny, dtype=float)[:, None].repeat(nx, axis=1)
            return X, Y

    def on_add_line(self):
        self.plot.add_line()
        self.update_profiles()

    def on_remove_line(self):
        self.plot.remove_line()
        self.update_profiles()

    def on_update_spectrum(self):
        self.update_profiles()

    def on_save_spectrum(self):
        # Recompute now for up-to-date plots
        self.update_profiles()
        if not self._last_profiles:
            QMessageBox.information(self, "No data", "No spectra to save. Add/move lines to generate profiles.")
            return
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
        self.plot.set_bandwidth(self.rxes_panel.bandwidth_ev())
        self.update_profiles()

    def update_profiles(self):
        self._last_profiles = []
        if self.dataset is None or self.dataset.kind != "2D":
            return
        ds = self.dataset
        Z = np.asarray(ds.z, dtype=float)
        X2D, Y2D = self._build_coordinate_grids(ds)
        incident_view = (ds.x2d is None or ds.y2d is None)
        width_ev = self.rxes_panel.bandwidth_ev()

        line_vals = self.plot.get_line_positions()
        if len(line_vals) == 0:
            self.plot.plot_profiles("", [])
            self.update_ui_state()
            return

        if self.rxes_panel.rb_extr_incident.isChecked():
            curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
            if incident_view:
                x_profile = ds.y if ds.y is not None else Y2D[:, 0]
                for val in line_vals:
                    mask = np.abs(X2D - val) <= (width_ev / 2.0)
                    y_profile = np.nansum(np.where(mask, Z, 0.0), axis=1)
                    curves.append((x_profile, y_profile, f"Ω={val:.2f} eV"))
                xlab = "ω (eV)"
            else:
                # Energy transfer view: profile vs Δ (use Y2D rows)
                x_profile = np.nanmedian(Y2D, axis=1)
                for val in line_vals:
                    mask = np.abs(X2D - val) <= (width_ev / 2.0)
                    y_profile = np.nansum(np.where(mask, Z, 0.0), axis=1)
                    curves.append((x_profile, y_profile, f"Ω={val:.2f} eV"))
                xlab = "Ω − ω (eV)"
            self.plot.plot_profiles(xlab, curves)
            self._last_profiles = curves

        elif self.rxes_panel.rb_extr_emission.isChecked():
            if not incident_view:
                return
            x_profile = ds.x if ds.x is not None else X2D[0, :]
            curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
            for val in line_vals:
                mask = np.abs(Y2D - val) <= (width_ev / 2.0)
                y_profile = np.nansum(np.where(mask, Z, 0.0), axis=0)
                curves.append((x_profile, y_profile, f"ω={val:.2f} eV"))
            self.plot.plot_profiles("Ω (eV)", curves)
            self._last_profiles = curves

        else:
            if incident_view:
                self.rxes_panel.rb_mode_transfer.setChecked(True)
                return
            x_profile = ds.x2d[0, :] if ds.x2d is not None else X2D[0, :]
            curves: List[Tuple[np.ndarray, np.ndarray, str]] = []
            for val in line_vals:
                mask = np.abs(Y2D - val) <= (width_ev / 2.0)
                y_profile = np.nansum(np.where(mask, Z, 0.0), axis=0)
                curves.append((x_profile, y_profile, f"Δ={val:.2f} eV"))
            self.plot.plot_profiles("Ω (eV)", curves)
            self._last_profiles = curves

        self.update_ui_state()

    # ----- Load XES and normalise -----

    def on_load_xes(self):
        """
        Load an XES spectrum (.nxs or ASCII two-column) and select a region to compute area.
        Apply normalisation to the current RXES scan by dividing Z by the area.
        """
        if self.current_scan_number is None:
            QMessageBox.information(self, "No RXES loaded", "Load an RXES .nxs scan first.")
            return

        filters = [
            "XES spectrum (*.nxs *.txt *.dat *.csv)",
            "NeXus (*.nxs)",
            "ASCII (*.txt *.dat *.csv)",
            "All files (*)"
        ]
        path, _ = QFileDialog.getOpenFileName(self, "Load XES spectrum", "", ";;".join(filters))
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            use_upper = self.rxes_panel.rb_upper.isChecked()
            channel = "upper" if use_upper else "lower"

            # Use i20_loader for both .nxs and ASCII
            x, y = i20_loader.xes_from_path(path, channel=channel)

            # Show selection dialog
            from i20_xes.widgets.normalise_dialog import NormaliseDialog
            dlg = NormaliseDialog(x, y, parent=self, title=f"XES ({'Upper' if use_upper else 'Lower'})")
            if dlg.exec() != QDialog.Accepted:
                return
            area = dlg.selected_area()
            if not np.isfinite(area) or area <= 0:
                QMessageBox.warning(self, "Invalid area", "Selected XES area is not valid or non-positive.")
                return

            # Store norm factor on current scan
            self.scan[self.current_scan_number]["norm_factor"] = float(area)
            self.rxes_panel.lbl_norm.setText(f"Normalised by XES area: {area:.6g}")

            # Replot with normalisation applied
            self.refresh_rxes_view()
        except Exception as e:
            QMessageBox.critical(self, "XES load error", f"Failed to load/normalise from XES:\n{path}\n\n{e}")

    def on_save_ascii(self):
        if self.dataset is None:
            QMessageBox.information(self, "No data", "Nothing to save. Load data first.")
            return

        # Pick a sensible default filename
        base = "data.csv" if self.dataset.kind == "2D" else "spectrum.csv"
        start = os.path.join(os.path.dirname(self.dataset.source), base) if self.dataset.source else base

        path, _ = QFileDialog.getSaveFileName(
            self, "Save as ASCII", start,
            "CSV (*.csv);;All files (*)"
        )
        if not path:
            return

        try:
            io_mod.save_ascii(path, self.dataset)  # writes UTF-8, supports 1D, 2D grid, curvilinear
            self.status.showMessage(f"Saved ASCII: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save ASCII:\n{path}\n\n{e}")


    def on_save_nexus(self):
        if self.dataset is None:
            QMessageBox.information(self, "No data", "Nothing to save. Load data first.")
            return
        if not io_mod.H5_AVAILABLE:
            QMessageBox.warning(self, "h5py missing", "Install h5py to save NeXus files: pip install h5py")
            return

        base = "data.nxs"
        start = os.path.join(os.path.dirname(self.dataset.source), base) if self.dataset.source else base

        path, _ = QFileDialog.getSaveFileName(
            self, "Save as NeXus (HDF5)", start,
            "NeXus/HDF5 (*.nxs *.h5 *.hdf5);;All files (*)"
        )
        if not path:
            return

        try:
            io_mod.save_nexus(path, self.dataset)
            self.status.showMessage(f"Saved NeXus: {path}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save error", f"Failed to save NeXus:\n{path}\n\n{e}")