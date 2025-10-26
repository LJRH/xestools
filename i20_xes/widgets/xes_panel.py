from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QListWidget, QListWidgetItem, QAbstractItemView
)

SPECIAL_ROLE = Qt.UserRole + 1  # mark special rows like 'average'/'average_bkgsub'

class XESPanel(QGroupBox):
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

        # File list
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
        self.btn_save_average.setEnabled(False)
        self.btn_save_avg_norm.setEnabled(False)
        row2.addWidget(self.btn_average)
        row2.addWidget(self.btn_save_average)
        row2.addWidget(self.btn_save_avg_norm)
        row2.addStretch(1)

        # Normalisation
        norm_box = QGroupBox("Normalisation")
        norm_layout = QVBoxLayout(norm_box)
        rown = QHBoxLayout()
        self.btn_load_xes = QPushButton("Load XES…")
        self.lbl_norm = QLabel("Average: no normalisation")
        rown.addWidget(self.btn_load_xes)
        rown.addStretch(1)
        norm_layout.addLayout(rown)
        norm_layout.addWidget(self.lbl_norm)

        # Background section
        bkg_box = QGroupBox("Background")
        bkg_layout = QVBoxLayout(bkg_box)
        rowb1 = QHBoxLayout()
        self.btn_load_wide = QPushButton("Load wide scan…")
        self.btn_bkg_extract = QPushButton("Background Extraction…")
        rowb1.addWidget(self.btn_load_wide)
        rowb1.addStretch(1)
        rowb1.addWidget(self.btn_bkg_extract)

        rowb2 = QHBoxLayout()
        self.btn_save_fit_log = QPushButton("Save fit log")
        self.btn_save_bkg_data = QPushButton("Save bkg extracted data")
        self.btn_save_fit_log.setEnabled(False)
        self.btn_save_bkg_data.setEnabled(False)
        rowb2.addWidget(self.btn_save_fit_log)
        rowb2.addWidget(self.btn_save_bkg_data)
        rowb2.addStretch(1)

        self.lbl_wide = QLabel("Wide scan: not loaded")

        bkg_layout.addLayout(rowb1)
        bkg_layout.addLayout(rowb2)
        bkg_layout.addWidget(self.lbl_wide)

        v.addWidget(det_box)
        v.addWidget(QLabel("Loaded XES scans (tick to include in average):"))
        v.addWidget(self.list)
        v.addLayout(row1)
        v.addLayout(row2)
        v.addWidget(norm_box)
        v.addWidget(bkg_box)
        v.addStretch(1)

    # ----- list helpers -----

    def add_item(self, label: str, checked: bool = True):
        it = QListWidgetItem(label)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.list.addItem(it)

    def checked_indices(self):
        idxs = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            # ignore special rows in checked list (so they are NOT used for averaging)
            if it.data(SPECIAL_ROLE):
                continue
            if it.checkState() == Qt.CheckState.Checked:
                idxs.append(i)
        if idxs:
            return idxs
        # fallback to selected rows, but still ignore special rows
        return sorted({ix.row() for ix in self.list.selectedIndexes()
                       if not self.list.item(ix.row()).data(SPECIAL_ROLE)})
    
    def remove_selected_items(self):
        rows = sorted({i.row() for i in self.list.selectedIndexes()}, reverse=True)
        for r in rows:
            self.list.takeItem(r)

    def clear_items(self):
        self.list.clear()

    # ----- special rows ('average', 'average_bkgsub') -----

    def _find_special_row(self, key: str) -> int:
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(SPECIAL_ROLE) == key:
                return i
        return -1

    def upsert_special(self, key: str, label: str, checked: bool = True):
        """
        Create or replace a checkable row for special products:
          key in {'average', 'average_bksub'}
        These rows are checkable to show/hide in the plot, but are excluded
        from averaging thanks to checked_indices() filtering by SPECIAL_ROLE.
        """
        it = QListWidgetItem(label)
        it.setData(SPECIAL_ROLE, key)
        # checkable + selectable + enabled
        it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        # style hint: italic
        f = it.font(); f.setItalic(True); it.setFont(f)

        pos = self._find_special_row(key)
        if pos >= 0:
            self.list.takeItem(pos)
            self.list.insertItem(pos, it)
        else:
            self.list.addItem(it)

    def remove_special(self, key: str):
        pos = self._find_special_row(key)
        if pos >= 0:
            self.list.takeItem(pos)