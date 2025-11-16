from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QFormLayout, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QRadioButton, QSlider, QCheckBox, QSpinBox
)

class IOPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("I/O", parent)
        layout = QFormLayout(self)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)

        # Single Load button
        self.btn_load = QPushButton("Load…")

        self.btn_save_ascii = QPushButton("Save as ASCII…")
        self.btn_save_nexus = QPushButton("Save as NeXus…")

        self.info_label = QLabel("No data loaded")
        self.info_label.setWordWrap(True)

        # Status label for channel/mode/file
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

        # Detector selection: Upper / Lower
        det_box = QGroupBox("Detector channel")
        det_layout = QHBoxLayout(det_box)
        self.rb_upper = QRadioButton("Upper")
        self.rb_lower = QRadioButton("Lower")
        self.rb_upper.setChecked(True)
        self.rb_upper.setToolTip("Use ω from XESEnergyUpper and intensity from FFI1_medipix1")
        self.rb_lower.setToolTip("Use ω from XESEnergyLower and intensity from FFI1_medipix2")
        det_layout.addWidget(self.rb_upper)
        det_layout.addWidget(self.rb_lower)

        # Mode selection: Incident Energy vs Energy Transfer
        mode_box = QGroupBox("Display mode")
        mode_layout = QVBoxLayout(mode_box)
        mode_row1 = QHBoxLayout()
        self.rb_mode_incident = QRadioButton("Incident Energy")
        self.rb_mode_transfer = QRadioButton("Energy Transfer")
        self.rb_mode_incident.setChecked(True)
        self.rb_mode_incident.setToolTip("Plot Ω (X) vs ω (Y)")
        self.rb_mode_transfer.setToolTip("Plot Ω (X) vs (Ω − ω) (Y)")
        mode_row1.addWidget(self.rb_mode_incident)
        mode_row1.addWidget(self.rb_mode_transfer)
        mode_layout.addLayout(mode_row1)
        
        # Contour overlay controls
        mode_row2 = QHBoxLayout()
        self.chk_contours = QCheckBox("Show contours")
        self.chk_contours.setToolTip("Overlay contour lines on the RXES map")
        self.spn_contour_levels = QSpinBox()
        self.spn_contour_levels.setRange(3, 30)
        self.spn_contour_levels.setValue(10)
        self.spn_contour_levels.setToolTip("Number of contour levels")
        self.spn_contour_levels.setEnabled(False)
        mode_row2.addWidget(self.chk_contours)
        mode_row2.addWidget(QLabel("Levels:"))
        mode_row2.addWidget(self.spn_contour_levels)
        mode_row2.addStretch(1)
        mode_layout.addLayout(mode_row2)
        
        # Additional contour options
        mode_row3 = QHBoxLayout()
        from PySide6.QtWidgets import QComboBox, QDoubleSpinBox
        self.cmb_contour_color = QComboBox()
        self.cmb_contour_color.addItems(["white", "black", "red", "blue", "yellow", "cyan", "magenta", "gray"])
        self.cmb_contour_color.setToolTip("Contour line color")
        self.cmb_contour_color.setEnabled(False)
        self.spn_contour_width = QDoubleSpinBox()
        self.spn_contour_width.setRange(0.1, 5.0)
        self.spn_contour_width.setValue(0.5)
        self.spn_contour_width.setSingleStep(0.1)
        self.spn_contour_width.setToolTip("Contour line width")
        self.spn_contour_width.setEnabled(False)
        mode_row3.addWidget(QLabel("Color:"))
        mode_row3.addWidget(self.cmb_contour_color)
        mode_row3.addWidget(QLabel("Width:"))
        mode_row3.addWidget(self.spn_contour_width)
        mode_row3.addStretch(1)
        mode_layout.addLayout(mode_row3)
        
        # Contour display features (island, gravity, labels)
        mode_row4 = QHBoxLayout()
        self.chk_contour_fill = QCheckBox("Fill islands")
        self.chk_contour_fill.setToolTip("Fill closed contour regions with semi-transparent color")
        self.chk_contour_fill.setEnabled(False)
        self.chk_contour_gravity = QCheckBox("Show centers")
        self.chk_contour_gravity.setToolTip("Show gravity centers (centroids) of closed contours")
        self.chk_contour_gravity.setEnabled(False)
        self.chk_contour_labels = QCheckBox("Show values")
        self.chk_contour_labels.setToolTip("Label contours with their intensity values")
        self.chk_contour_labels.setEnabled(False)
        mode_row4.addWidget(self.chk_contour_fill)
        mode_row4.addWidget(self.chk_contour_gravity)
        mode_row4.addWidget(self.chk_contour_labels)
        mode_row4.addStretch(1)
        mode_layout.addLayout(mode_row4)
        
        # Connect checkbox to enable/disable all contour options
        def enable_contour_options(enabled):
            self.spn_contour_levels.setEnabled(enabled)
            self.cmb_contour_color.setEnabled(enabled)
            self.spn_contour_width.setEnabled(enabled)
            self.chk_contour_fill.setEnabled(enabled)
            self.chk_contour_gravity.setEnabled(enabled)
            self.chk_contour_labels.setEnabled(enabled)
        self.chk_contours.toggled.connect(enable_contour_options)

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
        self.sl_width.setValue(10)       # 2.0 eV default
        self.lbl_width = QLabel("2.0 eV")
        row3.addWidget(self.sl_width)
        row3.addWidget(self.lbl_width)

        roi_layout.addLayout(row1)
        roi_layout.addLayout(row2)
        roi_layout.addLayout(row3)

        v.addWidget(det_box)
        v.addWidget(mode_box)
        v.addWidget(norm_box)
        v.addWidget(roi_box)
        v.addStretch(1)

        # Label update
        self.sl_width.valueChanged.connect(lambda v: self.lbl_width.setText(f"{v/5:.1f} eV"))


class XESPanel(QGroupBox):
    """
    Simple XES panel:
      - Detector channel (Upper/Lower)
      - Normalisation (Load XES… + status label)
    No display mode or ROI extraction here.
    """
    def __init__(self, parent=None):
        super().__init__("XES", parent)
        v = QVBoxLayout(self)

        # Detector selection: Upper / Lower
        det_box = QGroupBox("Detector channel")
        det_layout = QHBoxLayout(det_box)
        self.rb_upper = QRadioButton("Upper")
        self.rb_lower = QRadioButton("Lower")
        self.rb_upper.setChecked(True)
        det_layout.addWidget(self.rb_upper)
        det_layout.addWidget(self.rb_lower)

        # Normalisation section
        norm_box = QGroupBox("Normalisation")
        norm_layout = QVBoxLayout(norm_box)
        rown = QHBoxLayout()
        self.btn_load_xes = QPushButton("Load XES…")
        self.lbl_norm = QLabel("No normalisation applied")
        rown.addWidget(self.btn_load_xes)
        rown.addStretch(1)
        norm_layout.addLayout(rown)
        norm_layout.addWidget(self.lbl_norm)

        v.addWidget(det_box)
        v.addWidget(norm_box)
        v.addStretch(1)