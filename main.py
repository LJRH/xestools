import sys
from PySide6.QtWidgets import QApplication
from i20_xes.main_gui import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()