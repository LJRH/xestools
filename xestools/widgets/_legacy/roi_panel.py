from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider
)

class ROIPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("ROI Extraction", parent)
        v = QVBoxLayout(self)

        # Extraction type
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Extract:"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems([
            "Constant Incident (Ω)",
            "Constant Emission (ω)",
            "Constant Transfer (Ω−ω)"
        ])
        self.cb_mode.setToolTip("Pick the quantity to keep constant for 1D extraction")
        h1.addWidget(self.cb_mode)
        h1.addStretch(1)

        # Width slider (band half-width in eV)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Band width (eV):"))
        self.sl_width = QSlider(Qt.Horizontal)
        self.sl_width.setMinimum(0)     # 1 eV
        self.sl_width.setMaximum(3)   # 3 eV
        self.sl_width.setValue(1)      # 1 eV default
        self.lbl_width = QLabel("1 eV")
        h2.addWidget(self.sl_width)
        h2.addWidget(self.lbl_width)

        v.addLayout(h1)
        v.addLayout(h2)
        v.addStretch(1)

        self.sl_width.valueChanged.connect(self._on_width)

    def _on_width(self, v):
        self.lbl_width.setText(f"{v} eV")

    def current_mode(self) -> str:
        return self.cb_mode.currentText()

    def current_width_ev(self) -> float:
        return float(self.sl_width.value())