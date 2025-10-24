# i20_xes/widgets/xes_panel.py

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QListWidget, QListWidgetItem, QAbstractItemView
)

class XESPanel(QGroupBox):
    """
    XES workflow:
      - Choose detector channel (Upper/Lower)
      - Load multiple XES scans (ASCII or .nxs)
      - Select/deselect scans for averaging (checkbox)
      - Average selected scans
      - Normalise the averaged spectrum (separate section, 'Load XES…' button)
      - Save average and save normalised average
    """
    def __init__(self, parent=None):
        super().__init__("XES", parent)
        v = QVBoxLayout(self)

        # Detector selection
        det_box = QGroupBox("Detector channel")
        det_layout = QHBoxLayout(det_box)
        self.rb_upper = QRadioButton("Upper")
        self.rb_lower = QRadioButton("Lower")
        self.rb_upper.setChecked(True)
        det_layout.addWidget(self.rb_upper)
        det_layout.addWidget(self.rb_lower)

        # File list (with checkboxes)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Row 1: load/remove/clear
        row1 = QHBoxLayout()
        self.btn_load_scans = QPushButton("Load scans…")
        self.btn_remove_selected = QPushButton("Remove selected")
        self.btn_clear_all = QPushButton("Clear all")
        row1.addWidget(self.btn_load_scans)
        row1.addWidget(self.btn_remove_selected)
        row1.addWidget(self.btn_clear_all)
        row1.addStretch(1)

        # Row 2: average + saves
        row2 = QHBoxLayout()
        self.btn_average = QPushButton("Average selected")
        self.btn_save_average = QPushButton("Save average")
        self.btn_save_avg_norm = QPushButton("Save normalised avg")
        self.btn_save_average.setEnabled(False)   # disabled until an average is created
        self.btn_save_avg_norm.setEnabled(False)  # disabled until normalised
        row2.addWidget(self.btn_average)
        row2.addWidget(self.btn_save_average)
        row2.addWidget(self.btn_save_avg_norm)
        row2.addStretch(1)

        # Normalisation section (separate)
        norm_box = QGroupBox("Normalisation")
        norm_layout = QVBoxLayout(norm_box)
        rown = QHBoxLayout()
        # Use the same button name as RXES ('Load XES…') to share code paths
        self.btn_load_xes = QPushButton("Load XES…")
        self.lbl_norm = QLabel("Average: no normalisation")
        rown.addWidget(self.btn_load_xes)
        rown.addStretch(1)
        norm_layout.addLayout(rown)
        norm_layout.addWidget(self.lbl_norm)

        v.addWidget(det_box)
        v.addWidget(QLabel("Loaded XES scans (tick to include in average):"))
        v.addWidget(self.list)
        v.addLayout(row1)
        v.addLayout(row2)
        v.addWidget(norm_box)
        v.addStretch(1)

    def add_item(self, label: str, checked: bool = True):
        it = QListWidgetItem(label)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.list.addItem(it)

    def checked_indices(self):
        # Primary: checkbox state; Fallback: selected rows
        idxs = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                idxs.append(i)
        if idxs:
            return idxs
        return sorted({ix.row() for ix in self.list.selectedIndexes()})

    def remove_selected_items(self):
        rows = sorted({i.row() for i in self.list.selectedIndexes()}, reverse=True)
        for r in rows:
            self.list.takeItem(r)

    def clear_items(self):
        self.list.clear()