from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QFormLayout, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QRadioButton, QSlider
)

class IOPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("I/O", parent)
        layout = QFormLayout(self)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)

        self.btn_load = QPushButton("Load…")
        self.btn_save_ascii = QPushButton("Save as ASCII…")
        self.btn_save_nexus = QPushButton("Save as NeXus…")

        self.info_label = QLabel("No data loaded")
        self.info_label.setWordWrap(True)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout.addRow(QLabel("Current file:"), self.path_edit)
        layout.addRow(self.btn_load)
        layout.addRow(self.btn_save_ascii)
        layout.addRow(self.btn_save_nexus)
        layout.addRow(QLabel("Info:"))
        layout.addRow(self.info_label)
        layout.addRow(QLabel("Status:"))
        layout.addRow(self.status_label)


class RXESPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("RXES", parent)
        v = QVBoxLayout(self)

        # Detector selection
        det_box = QGroupBox("Detector channel")
        det_layout = QHBoxLayout(det_box)
        self.rb_upper = QRadioButton("Upper")
        self.rb_lower = QRadioButton("Lower")
        self.rb_upper.setChecked(True)
        det_layout.addWidget(self.rb_upper)
        det_layout.addWidget(self.rb_lower)

        # Mode selection
        mode_box = QGroupBox("Display mode")
        mode_layout = QHBoxLayout(mode_box)
        self.rb_mode_incident = QRadioButton("Incident Energy")
        self.rb_mode_transfer = QRadioButton("Energy Transfer")
        self.rb_mode_incident.setChecked(True)
        mode_layout.addWidget(self.rb_mode_incident)
        mode_layout.addWidget(self.rb_mode_transfer)

        # Normalisation section (before ROI)
        norm_box = QGroupBox("Normalisation")
        norm_layout = QVBoxLayout(norm_box)
        rown = QHBoxLayout()
        self.btn_load_xes = QPushButton("Load XES…")
        self.lbl_norm = QLabel("No normalisation applied")
        rown.addWidget(self.btn_load_xes)
        rown.addStretch(1)
        norm_layout.addLayout(rown)
        norm_layout.addWidget(self.lbl_norm)

        # ROI Extraction section
        roi_box = QGroupBox("ROI Extraction")
        roi_layout = QVBoxLayout(roi_box)

        row1 = QHBoxLayout()
        self.rb_extr_incident = QRadioButton("Constant Incident (Ω)")
        self.rb_extr_emission = QRadioButton("Constant Emission (ω)")
        self.rb_extr_transfer = QRadioButton("Constant Transfer (Ω−ω)")
        self.rb_extr_incident.setChecked(True)
        row1.addWidget(self.rb_extr_incident)
        row1.addWidget(self.rb_extr_emission)
        row1.addWidget(self.rb_extr_transfer)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        self.btn_add_line = QPushButton("Add line")
        self.btn_remove_line = QPushButton("Remove line")
        self.btn_update_spectrum = QPushButton("Update Spectrum Plot")
        self.btn_save_spectrum = QPushButton("Save Spectrum")
        row2.addWidget(self.btn_add_line)
        row2.addWidget(self.btn_remove_line)
        row2.addStretch(1)
        row2.addWidget(self.btn_update_spectrum)
        row2.addWidget(self.btn_save_spectrum)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Band width (eV):"))
        self.sl_width = QSlider(Qt.Horizontal)
        self.sl_width.setMinimum(1)      # 0.2 eV
        self.sl_width.setMaximum(15)     # 3.0 eV
        self.sl_width.setSingleStep(1)   # 0.2 eV per step
        self.sl_width.setPageStep(1)
        self.sl_width.setValue(10)       # 2.0 eV
        self.lbl_width = QLabel("2.0 eV")
        row3.addWidget(self.sl_width)
        row3.addWidget(self.lbl_width)

        roi_layout.addLayout(row1)
        roi_layout.addLayout(row2)
        roi_layout.addLayout(row3)

        # Compose RXES panel
        v.addWidget(det_box)
        v.addWidget(mode_box)
        v.addWidget(norm_box)   # normalisation before ROI
        v.addWidget(roi_box)
        v.addStretch(1)

        # Small label updates
        self.sl_width.valueChanged.connect(lambda v: self.lbl_width.setText(f"{v/5:.1f} eV"))

    def bandwidth_ev(self) -> float:
        return self.sl_width.value() / 5.0

    def set_bandwidth_ev(self, width_ev: float):
        width_ev = max(0.2, min(3.0, float(width_ev)))
        self.sl_width.setValue(int(round(width_ev * 5)))